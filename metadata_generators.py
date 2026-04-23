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
        """Generate the fixed set of visibility entries for the dataset.
        
        Uses string integer tokens ("1"–"4") and level strings matching real NuScenes v1.0-mini:
          "1" -> "v0-40",  "2" -> "v40-60",  "3" -> "v60-80",  "4" -> "v80-100"
        """
        visibility_levels = [
            ("1", "v0-40",   "visibility of whole object is between 0 and 40%"),
            ("2", "v40-60",  "visibility of whole object is between 40 and 60%"),
            ("3", "v60-80",  "visibility of whole object is between 60 and 80%"),
            ("4", "v80-100", "visibility of whole object is between 80 and 100%"),
        ]
        self.converter.visibilities = []
        for token, level, description in visibility_levels:
            visibility_entry = {
                "token": token,
                "level": level,
                "description": description
            }
            self.converter.visibilities.append(visibility_entry)
            # Store token keyed by level string for annotation_generator lookup
            self.converter.token_maps['visibility'][level] = token