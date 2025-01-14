import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import spacy
from pathlib import Path
import json
import os
import asyncio
import textwrap

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain.chains import create_extraction_chain
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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
    async def from_pdf(cls, pdf_path: str, llm: ChatAnthropic) -> 'LegalDocument':
        """Create a LegalDocument instance from a PDF file with LangChain processing."""
        path = Path(pdf_path)
        
        # Load PDF using LangChain's loader
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        
        # Extract metadata
        metadata = {
            'filename': path.name,
            'pages': len(pages),
            'source': str(path)
        }
        
        # Combine page content
        content = "\n".join(page.page_content for page in pages)
        
        # Split into sections using LangChain's text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_text(content)
        
        # Create sections from splits
        sections = []
        for i, split in enumerate(splits):
            sections.append({
                "id": str(i),
                "title": f"Section {i+1}",
                "content": split,
                "page": i // 2  # Approximate page number
            })
            if i > 0:
                sections[-1]["previous_section"] = str(i-1)
        
        # Create extraction chain for document analysis
        extraction_chain = create_extraction_chain(
            schema={
                "properties": {
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["summary", "key_points"],
            },
            llm=llm
        )
        
        # Extract summary and key points
        try:
            extraction_response = await extraction_chain.ainvoke({
                "input": content[:10000]  # First 10k characters for overview
            })
            summary = extraction_response['summary']
            key_points = extraction_response['key_points']
        except Exception as e:
            print(f"Error in document analysis: {str(e)}")
            summary = None
            key_points = None
        
        return cls(
            id=path.stem,
            title=metadata.get('title', path.stem),
            content=content,
            metadata=metadata,
            sections=sections,
            summary=summary,
            key_points=key_points
        )

class LegalMultiGraphRAG:
    def __init__(self, anthropic_api_key: str):
        # Initialize LangChain components
        self.llm = ChatAnthropic(api_key=anthropic_api_key)
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize graphs
        self.citation_graph = nx.DiGraph()
        self.concept_graph = nx.Graph()
        self.section_graph = nx.DiGraph()
        
        # Initialize vector stores
        self.document_store = None
        self.section_store = None
        
        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")
        
        # Create LangChain chains
        self._create_analysis_chains()
    
    def _create_analysis_chains(self):
        """Create LangChain chains for various analysis tasks."""
        # Concept extraction chain
        concept_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert legal analyst. Extract key legal concepts."),
            ("user", "Extract key legal concepts from the following text. Return as JSON array:\n\n{text}")
        ])
        self.concept_chain = LLMChain(llm=self.llm, prompt=concept_prompt)
        
        # Citation extraction chain
        citation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert legal analyst. Extract legal citations."),
            ("user", "Extract all legal citations from the following text. Return as JSON array:\n\n{text}")
        ])
        self.citation_chain = LLMChain(llm=self.llm, prompt=citation_prompt)
        
        # Section analysis chain
        section_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert legal analyst. Analyze document sections."),
            ("user", "Provide a brief analysis of this legal document section:\n\n{text}")
        ])
        self.section_chain = LLMChain(llm=self.llm, prompt=section_prompt)
    
    async def process_pdf_directory(self, directory_path: str) -> None:
        """Process all PDF files in a directory using LangChain."""
        pdf_files = list(Path(directory_path).glob('*.pdf'))
        documents = []
        sections = []
        
        for pdf_path in pdf_files:
            try:
                document = await LegalDocument.from_pdf(str(pdf_path), self.llm)
                await self.process_document(document)
                
                # Prepare documents for vector store
                documents.append(
                    Document(
                        page_content=document.content,
                        metadata={
                            'id': document.id,
                            'title': document.title,
                            'summary': document.summary,
                            'key_points': document.key_points
                        }
                    )
                )
                
                # Prepare sections for vector store
                for section in document.sections:
                    sections.append(
                        Document(
                            page_content=section['content'],
                            metadata={
                                'id': f"{document.id}_s{section['id']}",
                                'document_id': document.id,
                                'title': section['title'],
                                'page': section['page']
                            }
                        )
                    )
                
                print(f"Successfully processed {pdf_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
        
        # Create vector stores
        self.document_store = FAISS.from_documents(documents, self.embeddings)
        self.section_store = FAISS.from_documents(sections, self.embeddings)
    
    async def process_document(self, document: LegalDocument) -> None:
        """Process a legal document and update all graphs using LangChain."""
        # Add document node to citation graph
        self.citation_graph.add_node(
            document.id, 
            title=document.title, 
            metadata=document.metadata,
            summary=document.summary,
            key_points=document.key_points
        )
        
        # Process sections
        await self._process_sections(document)
        
        # Extract and process concepts
        await self._process_concepts(document)
        
        # Process citations
        await self._process_citations(document)
    
    async def _process_sections(self, document: LegalDocument) -> None:
        """Process document sections using LangChain."""
        for section in document.sections:
            section_id = f"{document.id}_s{section['id']}"
            
            # Get section analysis
            try:
                analysis = await self.section_chain.ainvoke({
                    "text": section['content'][:5000]
                })
                section_analysis = analysis['text']
            except Exception as e:
                print(f"Error in section analysis: {str(e)}")
                section_analysis = None
            
            # Add to section graph
            self.section_graph.add_node(
                section_id,
                document_id=document.id,
                title=section['title'],
                content=section['content'],
                page=section.get('page', 0),
                analysis=section_analysis
            )
            
            # Connect sequential sections
            if 'previous_section' in section:
                prev_section_id = f"{document.id}_s{section['previous_section']}"
                self.section_graph.add_edge(prev_section_id, section_id, 
                                          relationship="sequential")
    
    async def _process_concepts(self, document: LegalDocument) -> None:
        """Extract legal concepts using LangChain."""
        try:
            response = await self.concept_chain.ainvoke({
                "text": document.content[:10000]
            })
            concepts = json.loads(response['text'])
            
            # Add to concept graph
            for concept in concepts:
                self.concept_graph.add_node(concept)
                self.concept_graph.add_edge(document.id, concept, 
                                          relationship="contains")
        except Exception as e:
            print(f"Error extracting concepts: {str(e)}")
    
    async def _process_citations(self, document: LegalDocument) -> None:
        """Process document citations using LangChain."""
        try:
            response = await self.citation_chain.ainvoke({
                "text": document.content[:10000]
            })
            citations = json.loads(response['text'])
            
            for citation in citations:
                cited_id = citation.lower().replace(" ", "_")
                self.citation_graph.add_edge(document.id, cited_id,
                                           citation_text=citation)
        except Exception as e:
            print(f"Error extracting citations: {str(e)}")
    
    async def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query the system using LangChain retrieval chain."""
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(
            create_stuff_documents_chain(
                llm=self.llm,
                prompt=ChatPromptTemplate.from_messages([
                    ("system", "You are an expert legal analyst."),
                    ("user", "Answer the following question based on the provided documents:\n\nQuestion: {input}\n\nDocuments: {context}")
                ])
            ),
            retriever=self.document_store.as_retriever(
                search_kwargs={"k": top_k}
            )
        )
        
        # Get relevant documents
        chain_response = await retrieval_chain.ainvoke({"input": query})
        
        # Process results
        results = []
        for i, doc in enumerate(chain_response["context"]):
            doc_id = doc.metadata['id']
            
            # Get relevant sections
            section_results = await self._get_relevant_sections(doc_id, query, top_k=3)
            
            # Calculate scores
            citation_score = self._calculate_citation_score(doc_id)
            concept_score = await self._calculate_concept_score(doc_id, query)
            
            results.append({
                'document_id': doc_id,
                'title': doc.metadata['title'],
                'summary': doc.metadata.get('summary'),
                'key_points': doc.metadata.get('key_points', []),
                'citation_score': citation_score,
                'concept_score': concept_score,
                'relevant_sections': section_results
            })
        
        return results
    
    def _calculate_citation_score(self, doc_id: str) -> float:
        """Calculate importance score based on citation graph."""
        if doc_id not in self.citation_graph:
            return 0.0
        
        pagerank = nx.pagerank(self.citation_graph)
        return pagerank.get(doc_id, 0.0)
    
    async def _calculate_concept_score(self, doc_id: str, query: str) -> float:
        """Calculate concept similarity score."""
        if doc_id not in self.concept_graph:
            return 0.0
        
        try:
            # Extract concepts from query
            query_concepts_response = await self.concept_chain.ainvoke({"text": query})
            query_concepts = set(json.loads(query_concepts_response['text']))
            
            # Get document concepts
            doc_concepts = set(self.concept_graph.neighbors(doc_id))
            
            # Calculate Jaccard similarity
            if not query_concepts or not doc_concepts:
                return 0.0
            
            return len(query_concepts & doc_concepts) / len(query_concepts | doc_concepts)
            
        except Exception as e:
            print(f"Error calculating concept score: {str(e)}")
            return 0.0
    
    async def _get_relevant_sections(self, doc_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """Get relevant sections using LangChain retrieval."""
        # Get sections from vector store
        section_results = self.section_store.similarity_search_with_score(
            query,
            k=top_k,
            filter={"document_id": doc_id}
        )
        
        relevant_sections = []
        for section, score in section_results:
            section_id = section.metadata['id']
            section_data = self.section_graph.nodes[section_id]
            
            relevant_sections.append({
                'section_id': section_id,
                'title': section_data['title'],
                'content': section_data['content'],
                'page': section_data['page'],
                'analysis': section_data.get('analysis', ''),
                'score': float(score)
            })
        
        return sorted(relevant_sections, key=lambda x: x['score'], reverse=True)

async def process_legal_documents(pdf_directory: str, anthropic_api_key: str) -> None:
    """Helper function to process legal documents and perform queries."""
    rag = LegalMultiGraphRAG(anthropic_api_key=anthropic_api_key)
    await rag.process_pdf_directory(pdf_directory)
    return rag

# Example usage
if __name__ == "__main__":
    async def main():
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
        
        rag = await process_legal_documents("./legal_docs", api_key)
        
        results = await rag.query("What are the compliance requirements for data privacy?")
        
        print("\nSearch Results:")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document: {result['document_id']}")
            print(f"Citation Score: {result['citation_score