import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import spacy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import re
from pathlib import Path

@dataclass
class LegalDocument:
    id: str
    title: str
    content: str
    metadata: Dict
    sections: List[Dict]
    
    @classmethod
    def from_pdf(cls, pdf_path: str) -> 'LegalDocument':
        """Create a LegalDocument instance from a PDF file."""
        path = Path(pdf_path)
        
        with open(path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            metadata = {
                'filename': path.name,
                'pages': len(pdf_reader.pages),
                'author': pdf_reader.metadata.get('/Author', ''),
                'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
                'modified_date': pdf_reader.metadata.get('/ModDate', '')
            }
            
            # Extract content and identify sections
            content = []
            sections = []
            current_section = {"content": [], "title": ""}
            section_id = 0
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                content.append(text)
                
                # Attempt to identify sections using common patterns
                lines = text.split('\n')
                for line in lines:
                    # Check for section headers (simplified pattern)
                    section_match = re.match(r'^(?:Section|Article|Part)\s+(\d+|[IVX]+)[.:]\s*(.+)', line)
                    if section_match:
                        # Save previous section if it exists
                        if current_section["content"]:
                            sections.append({
                                "id": str(section_id),
                                "title": current_section["title"],
                                "content": "\n".join(current_section["content"]),
                                "page": page_num
                            })
                            if section_id > 0:
                                sections[-1]["previous_section"] = str(section_id - 1)
                            section_id += 1
                        
                        # Start new section
                        current_section = {
                            "content": [],
                            "title": section_match.group(2).strip()
                        }
                    else:
                        current_section["content"].append(line)
            
            # Add the last section
            if current_section["content"]:
                sections.append({
                    "id": str(section_id),
                    "title": current_section["title"] or "Unnamed Section",
                    "content": "\n".join(current_section["content"]),
                    "page": len(pdf_reader.pages) - 1
                })
                if section_id > 0:
                    sections[-1]["previous_section"] = str(section_id - 1)
            
            return cls(
                id=path.stem,  # Use filename without extension as ID
                title=metadata.get('title', path.stem),
                content="\n".join(content),
                metadata=metadata,
                sections=sections
            )

class LegalMultiGraphRAG:
    def __init__(self):
        # Initialize different graphs for various relationships
        self.citation_graph = nx.DiGraph()  # For document citations
        self.concept_graph = nx.Graph()     # For shared legal concepts
        self.section_graph = nx.DiGraph()   # For section relationships
        
        # Initialize models
        self.nlp = spacy.load("en_core_web_sm")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Store document embeddings
        self.document_embeddings = {}
        self.section_embeddings = {}
        
    def process_pdf_directory(self, directory_path: str) -> None:
        """Process all PDF files in a directory."""
        pdf_files = Path(directory_path).glob('*.pdf')
        for pdf_path in pdf_files:
            try:
                document = LegalDocument.from_pdf(str(pdf_path))
                self.process_document(document)
                print(f"Successfully processed {pdf_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
    
    def process_document(self, document: LegalDocument) -> None:
        """Process a legal document and update all graphs."""
        # Add document node to citation graph
        self.citation_graph.add_node(document.id, 
                                   title=document.title, 
                                   metadata=document.metadata)
        
        # Generate document embedding
        doc_embedding = self.encoder.encode(document.content)
        self.document_embeddings[document.id] = doc_embedding
        
        # Process sections
        self._process_sections(document)
        
        # Extract and process concepts
        self._process_concepts(document)
        
        # Find citations and update citation graph
        self._process_citations(document)
    
    def _extract_citation_patterns(self, text: str) -> List[str]:
        """Extract potential legal citations using regex patterns."""
        patterns = [
            r'\d+\s+U\.S\.\C\.\s+§\s+\d+',  # U.S. Code citations
            r'\d+\s+CFR\s+§\s+\d+',         # Code of Federal Regulations
            r'\d+\s+Fed\.\s+Reg\.\s+\d+',   # Federal Register
            r'Public\s+Law\s+\d+-\d+',      # Public Laws
            r'\d+\s+Stat\.\s+\d+'           # Statutes at Large
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            citations.extend(match.group() for match in matches)
        
        return citations

    def _process_sections(self, document: LegalDocument) -> None:
        """Process document sections and create section relationships."""
        for section in document.sections:
            section_id = f"{document.id}_s{section['id']}"
            
            # Add section node to section graph
            self.section_graph.add_node(section_id,
                                      document_id=document.id,
                                      title=section['title'],
                                      content=section['content'],
                                      page=section.get('page', 0))
            
            # Generate section embedding
            section_embedding = self.encoder.encode(section['content'])
            self.section_embeddings[section_id] = section_embedding
            
            # Connect sequential sections
            if 'previous_section' in section:
                prev_section_id = f"{document.id}_s{section['previous_section']}"
                self.section_graph.add_edge(prev_section_id, section_id, 
                                          relationship="sequential")
    
    def _process_concepts(self, document: LegalDocument) -> None:
        """Extract legal concepts and create concept relationships."""
        doc = self.nlp(document.content)
        
        # Extract legal entities and concepts
        legal_concepts = set()
        for ent in doc.ents:
            if ent.label_ in ["LAW", "ORG", "GPE"]:
                legal_concepts.add(ent.text)
        
        # Add concepts to concept graph
        for concept in legal_concepts:
            self.concept_graph.add_node(concept)
            self.concept_graph.add_edge(document.id, concept, 
                                      relationship="contains")
    
    def _process_citations(self, document: LegalDocument) -> None:
        """Process document citations and update citation graph."""
        # Extract citations using patterns
        citations = self._extract_citation_patterns(document.content)
        
        for citation in citations:
            # Create a normalized ID for the citation
            cited_id = re.sub(r'\s+', '_', citation.lower())
            
            # Add to citation graph
            self.citation_graph.add_edge(document.id, cited_id,
                                       citation_text=citation)

    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query the multi-graph system to find relevant documents and sections."""
        query_embedding = self.encoder.encode(query)
        
        # Get document similarities
        doc_similarities = {}
        for doc_id, embedding in self.document_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            doc_similarities[doc_id] = similarity
        
        # Get section similarities
        section_similarities = {}
        for section_id, embedding in self.section_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            section_similarities[section_id] = similarity
        
        # Combine similarities with graph-based features
        results = self._rank_results(query, doc_similarities, section_similarities)
        
        return results[:top_k]
    
    def _rank_results(self, query: str, 
                     doc_similarities: Dict[str, float],
                     section_similarities: Dict[str, float]) -> List[Dict]:
        """Rank results combining embedding similarities and graph features."""
        results = []
        
        for doc_id, similarity in doc_similarities.items():
            # Calculate graph-based features
            citation_score = self._calculate_citation_score(doc_id)
            concept_score = self._calculate_concept_score(doc_id, query)
            
            # Combine scores
            final_score = (
                0.4 * similarity +          # Direct content similarity
                0.3 * citation_score +      # Citation importance
                0.3 * concept_score         # Concept relevance
            )
            
            # Get relevant sections
            relevant_sections = self._get_relevant_sections(
                doc_id, section_similarities)
            
            # Get metadata
            metadata = self.citation_graph.nodes[doc_id].get('metadata', {})
            
            results.append({
                'document_id': doc_id,
                'score': final_score,
                'similarity': similarity,
                'citation_score': citation_score,
                'concept_score': concept_score,
                'relevant_sections': relevant_sections,
                'metadata': metadata
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _calculate_citation_score(self, doc_id: str) -> float:
        """Calculate importance score based on citation graph."""
        if doc_id not in self.citation_graph:
            return 0.0
        
        # Use PageRank to calculate citation importance
        pagerank = nx.pagerank(self.citation_graph)
        return pagerank.get(doc_id, 0.0)
    
    def _calculate_concept_score(self, doc_id: str, query: str) -> float:
        """Calculate relevance score based on shared concepts."""
        if doc_id not in self.concept_graph:
            return 0.0
        
        # Get concepts from query
        query_doc = self.nlp(query)
        query_concepts = set(ent.text for ent in query_doc.ents 
                           if ent.label_ in ["LAW", "ORG", "GPE"])
        
        # Get document concepts
        doc_concepts = set(
            node for node in self.concept_graph.neighbors(doc_id)
        )
        
        # Calculate Jaccard similarity between concept sets
        if not query_concepts or not doc_concepts:
            return 0.0
            
        return len(query_concepts & doc_concepts) / len(query_concepts | doc_concepts)
    
    def _get_relevant_sections(self, doc_id: str, 
                             section_similarities: Dict[str, float],
                             top_k: int = 3) -> List[Dict]:
        """Get most relevant sections for a document."""
        doc_sections = [
            (section_id, score) for section_id, score in section_similarities.items()
            if section_id.startswith(f"{doc_id}_s")
        ]
        
        doc_sections.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                'section_id': section_id,
                'score': score,
                'content': self.section_graph.nodes[section_id]['content'],
                'page': self.section_graph.nodes[section_id]['page']
            }
            for section_id, score in doc_sections[:top_k]
        ]

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = LegalMultiGraphRAG()
    
    # Process a directory of PDF documents
    rag.process_pdf_directory("./data")

    
    while True:
        q = input("Ask a question: ")
        if q == "exit":
            break
        if q == "print":   
            print_messages(messages)
            continue

        # Query the system
        results = rag.query("Find regulations about compliance requirements")
        
        # Print results
        for result in results:
            print(f"\nDocument: {result['document_id']}")
            print(f"Score: {result['score']:.3f}")
            print(f"Metadata: {result['metadata']}")
            print("\nRelevant Sections:")
            for section in result['relevant_sections']:
                print(f"\n  - Page {section['page']}")
                print(f"    Score: {section['score']:.3f}")
                print(f"    Content preview: {section['content'][:200]}...")
    
    