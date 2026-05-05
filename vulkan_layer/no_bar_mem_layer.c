/*
 * VK_LAYER_MUSE_no_bar_mem
 *
 * Strips VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT from any memory type that is
 * also HOST_VISIBLE.  On systems where ReBAR is disabled at the driver level
 * the HOST_VISIBLE|DEVICE_LOCAL heap (MemoryTypeIndex=1) is only 32 MB, yet
 * UE5 tries to allocate 16-18 MB from it, quickly exhausting the heap and
 * crashing.  Removing DEVICE_LOCAL from those types forces UE5's Vulkan RHI
 * to fall back to the plain HOST_VISIBLE heap (system RAM) instead.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o no_bar_mem_layer.so no_bar_mem_layer.c \
 *       -ldl -Wl,-Bsymbolic
 */

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_XLIB_KHR

#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <stdlib.h>

/* ---- Minimal Vulkan types we need ---------------------------------------- */
#define VK_DEFINE_HANDLE(obj) typedef struct obj##_T* obj;
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(obj) typedef uint64_t obj;

VK_DEFINE_HANDLE(VkInstance)
VK_DEFINE_HANDLE(VkPhysicalDevice)
VK_DEFINE_HANDLE(VkDevice)

typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef VkFlags  VkMemoryPropertyFlags;
typedef VkFlags  VkMemoryHeapFlags;

#define VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT   0x00000001
#define VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT   0x00000002
#define VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  0x00000004
#define VK_MEMORY_HEAP_DEVICE_LOCAL_BIT       0x00000001
#define VK_MAX_MEMORY_TYPES 32
#define VK_MAX_MEMORY_HEAPS 16
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 1000048006

typedef struct {
    VkMemoryPropertyFlags propertyFlags;
    uint32_t              heapIndex;
} VkMemoryType;

typedef struct {
    VkDeviceSize      size;
    VkMemoryHeapFlags flags;
} VkMemoryHeap;

typedef struct {
    uint32_t     memoryTypeCount;
    VkMemoryType memoryTypes[VK_MAX_MEMORY_TYPES];
    uint32_t     memoryHeapCount;
    VkMemoryHeap memoryHeaps[VK_MAX_MEMORY_HEAPS];
} VkPhysicalDeviceMemoryProperties;

typedef void* VkStructureType_placeholder;
typedef struct {
    int      sType;   /* VkStructureType */
    void*    pNext;
    VkPhysicalDeviceMemoryProperties memoryProperties;
} VkPhysicalDeviceMemoryProperties2;

/* ---- Layer dispatch glue -------------------------------------------------- */
typedef void (*PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddr)(VkInstance, const char*);
typedef PFN_vkVoidFunction (*PFN_vkGetDeviceProcAddr)(VkDevice, const char*);
typedef void (*PFN_vkGetPhysicalDeviceMemoryProperties)(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
typedef void (*PFN_vkGetPhysicalDeviceMemoryProperties2)(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2*);
typedef void (*PFN_vkGetPhysicalDeviceMemoryProperties2KHR)(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2*);

/*
 * UE5's Vulkan loader calls vkGetInstanceProcAddr with a chain object as the
 * first arg (a VkInstance that is actually a layer dispatch table pointer).
 * We need to pass calls down the chain.
 */
typedef struct {
    PFN_vkGetInstanceProcAddr         GetInstanceProcAddr;
    PFN_vkGetPhysicalDeviceMemoryProperties  GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties2 GetPhysicalDeviceMemoryProperties2;
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR GetPhysicalDeviceMemoryProperties2KHR;
} LayerInstanceDispatch;

/* Single-slot dispatch table — we only expect one VkInstance at a time */
static LayerInstanceDispatch g_dispatch;
static int g_initialized = 0;

/* Strip DEVICE_LOCAL from types that are also HOST_VISIBLE */
static void patch_mem_props(VkPhysicalDeviceMemoryProperties *p)
{
    for (uint32_t i = 0; i < p->memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = p->memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            p->memoryTypes[i].propertyFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            /* Also clear DEVICE_LOCAL on the heap itself so size accounting is correct */
            uint32_t hi = p->memoryTypes[i].heapIndex;
            p->memoryHeaps[hi].flags &= ~VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
        }
    }
}

/* ---- Intercepted functions ----------------------------------------------- */
static void layer_GetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties *pProps)
{
    g_dispatch.GetPhysicalDeviceMemoryProperties(pd, pProps);
    patch_mem_props(pProps);
}

static void layer_GetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2 *pProps2)
{
    g_dispatch.GetPhysicalDeviceMemoryProperties2(pd, pProps2);
    patch_mem_props(&pProps2->memoryProperties);
}

static void layer_GetPhysicalDeviceMemoryProperties2KHR(
    VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2 *pProps2)
{
    g_dispatch.GetPhysicalDeviceMemoryProperties2KHR(pd, pProps2);
    patch_mem_props(&pProps2->memoryProperties);
}

/* ---- Layer entry points -------------------------------------------------- */

/* Forward declaration */
__attribute__((visibility("default")))
PFN_vkVoidFunction vkGetInstanceProcAddr_layer(VkInstance instance, const char *pName);

/*
 * vkNegotiateLoaderLayerInterfaceVersion — required by Vulkan loader >= 1.1
 */
typedef struct {
    uint32_t sType;               /* LAYER_NEGOTIATE_INTERFACE_STRUCT = 1 */
    void    *pNext;
    uint32_t loaderLayerInterfaceVersion;
    PFN_vkGetInstanceProcAddr   pfnGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr     pfnGetDeviceProcAddr;
    PFN_vkGetInstanceProcAddr   pfnGetPhysicalDeviceProcAddr;
} VkNegotiateLayerInterface;

#define LAYER_NEGOTIATE_INTERFACE_STRUCT 1

__attribute__((visibility("default")))
int32_t vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct)
{
    if (pVersionStruct->loaderLayerInterfaceVersion > 2)
        pVersionStruct->loaderLayerInterfaceVersion = 2;
    pVersionStruct->pfnGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr_layer;
    pVersionStruct->pfnGetDeviceProcAddr   = NULL;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
    return 0; /* VK_SUCCESS */
}

__attribute__((visibility("default")))
PFN_vkVoidFunction vkGetInstanceProcAddr_layer(VkInstance instance, const char *pName)
{
    if (!pName) return NULL;

    /*
     * On the first call the loader passes a special "chain" object as instance.
     * We use it to resolve the next layer's (or ICD's) function pointers.
     */
    if (instance && !g_initialized) {
        /* The chain object starts with a pointer to GetInstanceProcAddr */
        PFN_vkGetInstanceProcAddr next_gipa =
            *(PFN_vkGetInstanceProcAddr*)(*(void**)instance);
        g_dispatch.GetInstanceProcAddr = next_gipa;

#define RESOLVE(fn) \
        g_dispatch.fn = (PFN_##vk##fn) next_gipa(instance, "vk" #fn); \

        RESOLVE(GetPhysicalDeviceMemoryProperties)
        RESOLVE(GetPhysicalDeviceMemoryProperties2)
        RESOLVE(GetPhysicalDeviceMemoryProperties2KHR)
#undef RESOLVE
        g_initialized = 1;
    }

    if (strcmp(pName, "vkGetPhysicalDeviceMemoryProperties") == 0)
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceMemoryProperties;
    if (strcmp(pName, "vkGetPhysicalDeviceMemoryProperties2") == 0)
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceMemoryProperties2;
    if (strcmp(pName, "vkGetPhysicalDeviceMemoryProperties2KHR") == 0)
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceMemoryProperties2KHR;
    if (strcmp(pName, "vkGetInstanceProcAddr") == 0)
        return (PFN_vkVoidFunction)vkGetInstanceProcAddr_layer;

    /* Pass everything else down the chain */
    if (g_dispatch.GetInstanceProcAddr)
        return g_dispatch.GetInstanceProcAddr(instance, pName);
    return NULL;
}
