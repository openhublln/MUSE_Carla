from nuscene_utils import generate_token

class MetadataGenerators:
    def __init__(self, converter):
        self.converter = converter

    def generate_category_entries(self):
        """Generate category.json entries based on category mappings in config."""
        category_mappings = self.converter.config.get("category_mappings", {})
        self.converter.categories = []
        
        for carla_type, nuscene_name in category_mappings.items():
            token = generate_token()
            description = "A car" if nuscene_name == "vehicle.car" else ""
            # TODO: Add more descriptions for other categories
            
            category_entry = {
                "token": token,
                "name": nuscene_name,
                "description": description,
                "index": None  # TODO later
            }
            self.converter.categories.append(category_entry)

    def generate_attribute_entries(self):
        """Generate attribute.json entries based on config."""
        attribute_definitions = self.converter.config.get("attributes", {})
        self.converter.attributes = []
        
        # Process vehicle attributes
        if "vehicle" in attribute_definitions:
            for attr in attribute_definitions["vehicle"]:
                token = generate_token()
                attribute_entry = {
                    "token": token,
                    "name": attr["name"],
                    "description": f"Vehicle state: {attr['name'].split('.')[-1]}"
                }
                self.converter.attributes.append(attribute_entry)
                # Store token for later use
                self.converter.attribute_name_to_token[attr["name"]] = token
        
    def generate_visibility_entries(self):
        """Generate the fixed set of visibility entries for the dataset."""
        visibility_levels = [
            ("v0", "0-40%"),
            ("v1", "40-60%"),
            ("v2", "60-80%"),
            ("v3", "80-100%")
        ]
        self.converter.visibilities = []
        # Generate entries for each level
        for level, description in visibility_levels:
            token = generate_token()
            visibility_entry = {
                "token": token,
                "level": level,
                "description": description
            }
            self.converter.visibilities.append(visibility_entry)
            
            # Store token for later use
            self.converter.token_maps['visibility'][level] = token 