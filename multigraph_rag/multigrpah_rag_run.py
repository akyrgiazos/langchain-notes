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
from anthropic import Anthropic
import json
import os
from typing import List, Dict, Any
import asyncio
from anthropic.async_api import AsyncAnthropic
import textwrap

@dataclass
class LegalDocument:
    id: str
    title: str
    content: str
    metadata: Dict
    sections: List[Dict]
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    
    @classmethod
    async def from_pdf(cls, pdf_path: str, anthropic_client: AsyncAnthropic) -> 'LegalDocument':
        """Create a LegalDocument instance from a PDF file with Claude analysis."""
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

            full_content = "\n".join(content)

            # Use Claude to generate summary and key points
            try:
                summary_message = await anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    temperature=0,
                    system="You are an expert legal analyst. Provide concise, accurate summaries of legal documents.",
                    messages=[{
                        "role": "user",
                        "content": f"Please provide a concise summary (max 200 words) of this legal document:\n\n{full_content[:10000]}"
                    }]
                )
                summary = summary_message.content[0].text

                key_points_message = await anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    temperature=0,
                    system="You are an expert legal analyst. Extract key legal points and requirements.",
                    messages=[{
                        "role": "user",
                        "content": f"List the 5 most important legal points or requirements from this document:\n\n{full_content[:10000]}"
                    }]
                )
                key_points = key_points_message.content[0].text.split('\n')

            except Exception as e:
                print(f"Error getting Claude analysis: {str(e)}")
                summary = None
                key_points = None
            
            return cls(
                id=path.stem,
                title=metadata.get('title', path.stem),
                content=full_content,
                metadata=metadata,
                sections=sections,
                summary=summary,
                key_points=key_points
            )

class LegalMultiGraphRAG:
    def __init__(self, anthropic_api_key: str):
        # Initialize different graphs for various relationships
        self.citation_graph = nx.DiGraph()
        self.concept_graph = nx.Graph()
        self.section_graph = nx.DiGraph()
        
        # Initialize models
        self.nlp = spacy.load("en_core_web_sm")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Claude client
        self.anthropic = AsyncAnthropic(api_key=anthropic_api_key)
        
        # Store document embeddings
        self.document_embeddings = {}
        self.section_embeddings = {}
        
    async def process_pdf_directory(self, directory_path: str) -> None:
        """Process all PDF files in a directory."""
        pdf_files = list(Path(directory_path).glob('*.pdf'))
        for pdf_path in pdf_files:
            try:
                document = await LegalDocument.from_pdf(str(pdf_path), self.anthropic)
                await self.process_document(document)
                print(f"Successfully processed {pdf_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
    
    async def process_document(self, document: LegalDocument) -> None:
        """Process a legal document and update all graphs."""
        # Add document node to citation graph with Claude analysis
        self.citation_graph.add_node(
            document.id, 
            title=document.title, 
            metadata=document.metadata,
            summary=document.summary,
            key_points=document.key_points
        )
        
        # Generate document embedding
        doc_embedding = self.encoder.encode(document.content)
        self.document_embeddings[document.id] = doc_embedding
        
        # Process sections with Claude analysis
        await self._process_sections(document)
        
        # Extract and process concepts
        await self._process_concepts(document)
        
        # Find citations and update citation graph
        await self._process_citations(document)

    async def _process_sections(self, document: LegalDocument) -> None:
        """Process document sections and create section relationships."""
        for section in document.sections:
            section_id = f"{document.id}_s{section['id']}"
            
            # Get Claude's analysis of the section
            try:
                section_analysis = await self.anthropic.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=512,
                    temperature=0,
                    system="You are an expert legal analyst. Analyze document sections concisely.",
                    messages=[{
                        "role": "user",
                        "content": f"Provide a brief analysis of this legal document section:\n\n{section['content'][:5000]}"
                    }]
                )
                section_summary = section_analysis.content[0].text
            except Exception as e:
                print(f"Error getting section analysis: {str(e)}")
                section_summary = None
            
            # Add section node to section graph
            self.section_graph.add_node(
                section_id,
                document_id=document.id,
                title=section['title'],
                content=section['content'],
                page=section.get('page', 0),
                analysis=section_summary
            )
            
            # Generate section embedding
            section_embedding = self.encoder.encode(section['content'])
            self.section_embeddings[section_id] = section_embedding
            
            # Connect sequential sections
            if 'previous_section' in section:
                prev_section_id = f"{document.id}_s{section['previous_section']}"
                self.section_graph.add_edge(prev_section_id, section_id, 
                                          relationship="sequential")
    
    async def _process_concepts(self, document: LegalDocument) -> None:
        """Extract legal concepts using Claude and create concept relationships."""
        try:
            concept_analysis = await self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=512,
                temperature=0,
                system="You are an expert legal analyst. Extract key legal concepts.",
                messages=[{
                    "role": "user",
                    "content": f"List the key legal concepts and entities from this text. Format as JSON array:\n\n{document.content[:10000]}"
                }]
                )
            concepts = json.loads(concept_analysis.content[0].text)
        except Exception as e:
            print(f"Error extracting concepts: {str(e)}")
            concepts = []
        
        # Add concepts to concept graph
        for concept in concepts:
            self.concept_graph.add_node(concept)
            self.concept_graph.add_edge(document.id, concept, 
                                      relationship="contains")
    
    async def _process_citations(self, document: LegalDocument) -> None:
        """Process document citations using Claude and update citation graph."""
        try:
            citation_analysis = await self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=512,
                temperature=0,
                system="You are an expert legal analyst. Extract legal citations.",
                messages=[{
                    "role": "user",
                    "content": f"Extract all legal citations from this text. Format as JSON array:\n\n{document.content[:10000]}"
                }]
            )
            citations = json.loads(citation_analysis.content[0].text)
        except Exception as e:
            print(f"Error extracting citations: {str(e)}")
            citations = self._extract_citation_patterns(document.content)
        
        for citation in citations:
            cited_id = re.sub(r'\s+', '_', str(citation).lower())
            self.citation_graph.add_edge(document.id, cited_id,
                                       citation_text=citation)

    async def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query the multi-graph system with Claude enhancement."""
        # Get Claude's interpretation of the query
        try:
            query_analysis = await self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=512,
                temperature=0,
                system="You are an expert legal analyst. Help interpret legal queries.",
                messages=[{
                    "role": "user",
                    "content": f"Interpret this legal query and identify key concepts and requirements:\n\n{query}"
                }]
            )
            enhanced_query = query_analysis.content[0].text
        except Exception as e:
            print(f"Error analyzing query: {str(e)}")
            enhanced_query = query
        
        # Get embeddings for both original and enhanced queries
        query_embedding = self.encoder.encode(query)
        enhanced_query_embedding = self.encoder.encode(enhanced_query)
        
        # Calculate similarities using both embeddings
        doc_similarities = {}
        for doc_id, embedding in self.document_embeddings.items():
            orig_similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            enhanced_similarity = cosine_similarity([enhanced_query_embedding], [embedding])[0][0]
            # Use weighted average of similarities
            doc_similarities[doc_id] = 0.7 * orig_similarity + 0.3 * enhanced_similarity
        
        section_similarities = {}
        for section_id, embedding in self.section_embeddings.items():
            orig_similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            enhanced_similarity = cosine_similarity([enhanced_query_embedding], [embedding])[0][0]
            section_similarities[section_id] = 0.7 * orig_similarity + 0.3 * enhanced_similarity
        
        results = await self._rank_results(query, enhanced_query, doc_similarities, section_similarities)
        
        return results[:top_k]
    
    async def _rank_results(self, original_query: str, enhanced_query: str,
                          doc_similarities: Dict[str, float],
                          section_similarities: Dict[str, float]) -> List[Dict]:
        """Rank results with Claude enhancement."""
        results = []
        
        for doc_id, similarity in doc_similarities.items():
            # Calculate graph-based features
            citation_score = self._calculate_citation_score(doc_id)
            concept_score = await self._calculate_concept_score(doc_id, enhanced_query)
            
            # Get document metadata and Claude analysis
            metadata = self.citation_graph.nodes[doc_id].get('metadata', {})
            summary = self.citation_graph.nodes[doc_id].get('summary', '')
            key_points = self.citation_graph.nodes[doc_id].get('key_points', [])
            
            # Get relevant sections
            relevant_sections = await self._get_relevant_sections(
                doc_id, section_similarities, original_query)
            
            # Calculate final score
            final_score = (
                0.4 * similarity +          # Content similarity
                0.3 * citation_score +      # Citation importance
                0.3 * concept_score         # Concept relevance
            )
            
            results.append({
                'document_id': doc_id,
                'score': final_score,
                'similarity': similarity,
                'citation_score': citation_score,
                'concept_score': concept_score,
                'relevant_sections': relevant_sections,
                'metadata': metadata,
                'summary': summary,
                'key_points': key_points
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def _calculate_citation_score(self, doc_id: str) -> float:
        """Calculate importance score based on citation graph."""
        if doc_id not in self.citation_graph:
            return 0.0
        
        pagerank = nx.pagerank(self.citation_graph)
        return pagerank.get(doc_id, 0.0)
    
    async def _calculate_concept_score(self, doc_id: str, query: str) -> float:
        """Calculate relevance score based on shared concepts with Claude enhancement."""
        if doc_id not in self.concept_graph:
            return 0.0
        
        try:
            # Get Claude's analysis of conceptual similarity
            concept_analysis = await self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=512,
                temperature=0,
                system="You are an expert legal analyst. Analyze concept similarity.",
                messages=[{
                    "role": "user",
                    "content": f"Rate the conceptual similarity (0-1) between these legal concepts and the query. Return just the number:\n\nQuery: {query}\nConcepts: {list(self.concept_graph.neighbors(doc_id))}"
                }]
            )
            
            try:
                concept_similarity = float(concept_analysis.content[0].text.strip())
            except ValueError:
                concept_similarity = 0.0
                
            return concept_similarity
            
        except Exception as e:
            print(f"Error calculating concept score: {str(e)}")
            # Fallback to basic Jaccard similarity
            query_doc = self.nlp(query)
            query_concepts = set(ent.text for ent in query_doc.ents 
                               if ent.label_ in ["LAW", "ORG", "GPE"])
            
            doc_concepts = set(
                node for node in self.concept_graph.neighbors(doc_id)
            )
            
            if not query_concepts or not doc_concepts:
                return 0.0
                
            return len(query_concepts & doc_concepts) / len(query_concepts | doc_concepts)
    
    async def _get_relevant_sections(self, doc_id: str, 
                                   section_similarities: Dict[str, float],
                                   query: str,
                                   top_k: int = 3) -> List[Dict]:
        """Get most relevant sections with Claude enhancement."""
        doc_sections = [
            (section_id, score) for section_id, score in section_similarities.items()
            if section_id.startswith(f"{doc_id}_s")
        ]
        
        doc_sections.sort(key=lambda x: x[1], reverse=True)
        relevant_sections = []
        
        for section_id, score in doc_sections[:top_k]:
            section_data = self.section_graph.nodes[section_id]
            
            # Get Claude's analysis of section relevance
            try:
                relevance_analysis = await self.anthropic.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=512,
                    temperature=0,
                    system="You are an expert legal analyst. Analyze section relevance.",
                    messages=[{
                        "role": "user",
                        "content": f"Rate how relevant (0-1) this section is to the query. Return just the number:\n\nQuery: {query}\nSection: {section_data['content'][:5000]}"
                    }]
                )
                
                try:
                    claude_score = float(relevance_analysis.content[0].text.strip())
                except ValueError:
                    claude_score = 0.0
                
                # Combine embedding similarity with Claude's analysis
                final_score = 0.7 * score + 0.3 * claude_score
                
            except Exception as e:
                print(f"Error getting section relevance: {str(e)}")
                final_score = score
            
            relevant_sections.append({
                'section_id': section_id,
                'score': final_score,
                'content': section_data['content'],
                'title': section_data['title'],
                'page': section_data['page'],
                'analysis': section_data.get('analysis', '')
            })
        
        return sorted(relevant_sections, key=lambda x: x['score'], reverse=True)

async def process_legal_documents(pdf_directory: str, anthropic_api_key: str) -> None:
    """Helper function to process legal documents and perform queries."""
    # Initialize the RAG system
    rag = LegalMultiGraphRAG(anthropic_api_key=anthropic_api_key)
    
    # Process all PDFs in the directory
    await rag.process_pdf_directory(pdf_directory)
    
    return rag

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Get API key from environment variable
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
        
        # Initialize and process documents
        rag = await process_legal_documents("./data", api_key)
        
        # Example query
        results = await rag.query("Πώς πρέπει να συμπεριφέρεται ένας εργαζόμενος?")
        
        # Print results
        print("\nSearch Results:")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document: {result['document_id']}")
            print(f"Score: {result['score']:.3f}")
            print("\nSummary:")
            print(textwrap.fill(result['summary'], width=80))
            
            print("\nKey Points:")
            for point in result['key_points']:
                print(f"- {point}")
            
            print("\nRelevant Sections:")
            for section in result['relevant_sections']:
                print(f"\n  - {section['title']} (Page {section['page']})")
                print(f"    Relevance Score: {section['score']:.3f}")
                print("    Analysis:")
                print(textwrap.fill(section['analysis'], width=80, initial_indent="    ", subsequent_indent="    "))
                print("\n    Content Preview:")
                preview = section['content'][:300] + "..." if len(section['content']) > 300 else section['content']
                print(textwrap.fill(preview, width=80, initial_indent="    ", subsequent_indent="    "))
    
    # Run the async main function
    asyncio.run(main())