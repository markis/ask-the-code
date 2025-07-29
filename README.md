# Ask the Code

[![PyPI - Version](https://img.shields.io/pypi/v/ask-the-code.svg)](https://pypi.org/project/ask-the-code)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ask-the-code.svg)](https://pypi.org/project/ask-the-code)

A CLI tool for asking questions about documentation in a repository using AI-powered semantic search and retrieval-augmented generation (RAG).

-----

## Features

- **Semantic Search**: Index and search through markdown documentation using ChromaDB and FlagEmbedding
- **AI-Powered Answers**: Get contextual answers to questions about your documentation using Ollama
- **Repository-Aware**: Automatically indexes documentation files in your repository
- **Multiple Commands**: Create indexes, search sources, and ask natural language questions
- **Reranking**: Uses FlagEmbedding reranker for improved search relevance

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Commands](#commands)
- [Configuration](#configuration)
- [License](#license)

## Installation

```console
pip install ask-the-code
```

## Usage

First, create an index of your documentation:

```console
ask create
```

Then ask questions about your documentation:

```console
ask "How do I configure the application?"
```

Or search for specific information:

```console
ask search "configuration settings"
```

## Commands

- `ask create [--repo PATH] [--glob PATTERN]` - Create and index the knowledge store
- `ask "question"` - Ask a question about the documentation
- `ask search "query"` - Search for sources matching a query
- `ask clean` - Clean up the data directory

### Options

- `--repo, -r PATH` - The repository path (defaults to current directory)
- `--glob, -g PATTERN` - The glob pattern to match files (defaults to `**/*.md`)

## Configuration

The tool uses Ollama for language model inference and requires:

- Ollama running locally or accessible via URL
- A compatible language model (configurable via settings)
- ChromaDB for vector storage
- FlagEmbedding for text embeddings and reranking

## License

`ask-the-code` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
