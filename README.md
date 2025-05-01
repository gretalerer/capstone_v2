# Hybrid RAG Implementation

This project implements a sophisticated Hybrid RAG (Retrieval-Augmented Generation) system that combines multiple RAG approaches for robust question-answering capabilities. The system integrates Self-RAG, Corrective-RAG (CRAG), and Multi-hop RAG techniques to provide accurate and contextually relevant answers.

## Project Structure

```
.
├── hybrid_rag.py          # Main implementation of the Hybrid RAG system
├── chain.py              # Basic chain implementation
├── requirements.txt      # Project dependencies
├── tools/               # Custom tools and utilities
│   └── nl2sql_tools/    # Natural Language to SQL conversion tools
├── eval_RAGBit/         # Evaluation framework for RAG systems
├── langgraph/           # LangGraph related configurations
└── .langgraph_api/      # LangGraph API configurations
```

## Features

- **Hybrid RAG Architecture**: Combines multiple RAG approaches for improved accuracy
- **Multi-hop Retrieval**: Capable of breaking down complex questions into sub-questions
- **Self-Reflection**: Implements self-assessment of answer quality
- **Gap Analysis**: Identifies and addresses knowledge gaps in responses
- **SQL Integration**: Supports querying structured data through BigQuery
- **Vector Search**: Utilizes ChromaDB for efficient document retrieval
- **Evaluation Framework**: Includes tools for measuring RAG performance

## Dependencies

The project uses several key libraries:
- LangChain and LangGraph for orchestration
- OpenAI's GPT models for language processing
- ChromaDB for vector storage
- Google BigQuery for structured data access
- Tavily for web search capabilities
- Various evaluation and utility libraries

See `requirements.txt` for the complete list of dependencies.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file with necessary API keys
   - Configure BigQuery credentials

## Usage

The main implementation is in `hybrid_rag.py`. The system can be used to:
- Process natural language questions
- Retrieve relevant information from multiple sources
- Generate comprehensive answers with self-reflection
- Perform gap analysis and multi-hop reasoning

## Evaluation

The `eval_RAGBit` directory contains tools for evaluating the RAG system's performance, including:
- Gap analysis tracking
- Retrieval metrics
- Performance evaluation functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- OpenAI for GPT models
- LangChain and LangGraph teams
- Google BigQuery
- Tavily 