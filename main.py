#!/usr/bin/env python3
import os
import click
from rich.console import Console
from rich.panel import Panel
from utils import load_documents, search_documents, build_term_frequency_index, search_tfidf

console = Console()

@click.group()
def cli():
    """TextSearchEngine - A command-line text search application."""
    pass

@cli.command()
@click.argument('query')
@click.option('--case-sensitive/--case-insensitive', default=False, help='Enable case-sensitive search')
def search(query, case_sensitive):
    """Search for text in documents and show a preview with the keyword highlighted."""
    try:
        # Get the documents directory path
        documents_dir = os.path.join(os.path.dirname(__file__), 'documents')
        
        # Load and search documents
        documents = load_documents(documents_dir)
        results = search_documents(documents, query, case_sensitive)
        
        # Display results
        if results:
            console.print(Panel.fit(
                f"Found {len(results)} matches for '{query}'",
                title="Search Results",
                border_style="green"
            ))
            for doc_name, matches in results.items():
                console.print(f"\n[bold blue]{doc_name}[/bold blue]")
                # Show preview with keyword highlighted
                preview = documents[doc_name][:100].replace('\n', ' ')
                if not case_sensitive:
                    # Highlight all case-insensitive matches
                    import re
                    pattern = re.compile(re.escape(query), re.IGNORECASE)
                    preview = pattern.sub(f"[reverse][yellow]\\g<0>[/yellow][/reverse]", preview)
                else:
                    preview = preview.replace(query, f"[reverse][yellow]{query}[/yellow][/reverse]")
                console.print(f"  Preview: {preview}")
                for match in matches:
                    console.print(f"  • {match}")
        else:
            console.print(Panel.fit(
                f"No matches found for '{query}'",
                title="Search Results",
                border_style="yellow"
            ))
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
def list_docs():
    """List all available documents."""
    try:
        documents_dir = os.path.join(os.path.dirname(__file__), 'documents')
        if not os.path.exists(documents_dir):
            console.print("[yellow]No documents directory found.[/yellow]")
            return
            
        files = [f for f in os.listdir(documents_dir) if f.endswith('.txt')]
        if files:
            console.print(Panel.fit(
                "\n".join(f"• {f}" for f in files),
                title="Available Documents",
                border_style="blue"
            ))
        else:
            console.print("[yellow]No text documents found.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
@click.argument('keyword')
def ranked_search(keyword):
    """Display a ranked list of matching documents with relevance scores using TF-IDF."""
    try:
        documents_dir = os.path.join(os.path.dirname(__file__), 'documents')
        documents = load_documents(documents_dir)
        tf_index = build_term_frequency_index(documents)
        ranked_results = search_tfidf(keyword, tf_index, documents)
        if ranked_results:
            console.print(Panel.fit(
                f"Ranked results for '{keyword}':",
                title="TF-IDF Search Results",
                border_style="green"
            ))
            for i, (filename, score) in enumerate(ranked_results, 1):
                console.print(f"{i}. [bold blue]{filename}[/bold blue] - Relevance Score: [yellow]{score:.4f}[/yellow]")
        else:
            console.print(Panel.fit(
                f"No relevant documents found for '{keyword}'",
                title="TF-IDF Search Results",
                border_style="yellow"
            ))
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == '__main__':
    cli() 