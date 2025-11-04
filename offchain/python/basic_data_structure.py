from dataclasses import dataclass, field
from eth_utils import keccak

class Document:
    def __init__(self, doc_id, doc_type, content, province, property_id):
        self.doc_id = doc_id
        self.doc_type = doc_type
        self.content = content
        self.province = province
        self.property_id = property_id
        # Create hierarchical identifier: Province.Property
        self.full_id = f"{province}.{property_id}"
        self.hash_hex = keccak(self.content.encode('utf-8')).hex()

    def __repr__(self):
        return f"Doc(id={self.doc_id}, type={self.doc_type}, property={self.full_id})"
    
@dataclass
class Property:
    """Represents a real estate property containing a list of Document objects."""
    property_id: str
    province: str
    documents: list[Document] = field(default_factory=list)