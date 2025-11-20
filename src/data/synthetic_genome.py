"""
Generate synthetic genomic data for testing.
"""

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def create_synthetic_genome(length=5_000_000, output_file='synthetic_genome.fasta', 
                            gc_content=0.36, seed=42):
    """
    Generate synthetic genome with realistic base composition.
    
    Args:
        length (int): Length of genome in base pairs
        output_file (str): Output FASTA file path
        gc_content (float): Target GC content (default: 0.36 for C. elegans-like)
        seed (int): Random seed for reproducibility
        
    Returns:
        str: Path to output file
        
    Example:
        >>> genome_file = create_synthetic_genome(
        ...     length=1_000_000,
        ...     output_file='test_genome.fasta',
        ...     gc_content=0.5
        ... )
    """
    np.random.seed(seed)
    
    # Calculate base probabilities from GC content
    gc_prob = gc_content / 2  # Split equally between G and C
    at_prob = (1 - gc_content) / 2  # Split equally between A and T
    
    bases = ['A', 'T', 'G', 'C']
    weights = [at_prob, at_prob, gc_prob, gc_prob]
    
    # Generate sequence
    sequence = ''.join(np.random.choice(bases, size=length, p=weights))
    
    # Create FASTA record
    record = SeqRecord(
        Seq(sequence),
        id="synthetic_chromosome",
        description=f"Synthetic {length/1e6:.1f}Mb genome | GC={gc_content:.2%}"
    )
    
    # Write to file
    SeqIO.write(record, output_file, "fasta")
    
    print(f"✓ Created synthetic genome: {output_file} ({length/1e6:.1f} Mb)")
    print(f"  Target GC content: {gc_content:.2%}")
    print(f"  Actual GC content: {(sequence.count('G') + sequence.count('C'))/len(sequence):.2%}")
    
    return output_file


if __name__ == '__main__':
    # Quick test
    create_synthetic_genome(
        length=1_000_000,
        output_file='test_genome.fasta'
    )