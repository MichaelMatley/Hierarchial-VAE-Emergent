"""
DNA sequence encoding utilities.

Provides methods for converting DNA sequences to numerical representations
and back. Supports one-hot encoding with proper handling of ambiguous bases.
"""

import numpy as np


class DNAEncoder:
    """
    Static methods for DNA sequence encoding/decoding.
    
    Encoding schemes:
        - One-hot: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
        - Ordinal: A=0, C=1, G=2, T=3
        - K-mer: Frequency distribution of k-length subsequences
    
    Ambiguous bases (N, R, Y, etc.) are handled by:
        - One-hot: All-zero column (masked)
        - Ordinal: Mapped to 0 (same as A)
    
    Example:
        >>> encoder = DNAEncoder()
        >>> sequence = "ATCGATCG"
        >>> encoded = encoder.one_hot_encode(sequence)
        >>> print(encoded.shape)  # (4, 8)
        >>> decoded = encoder.decode_one_hot(encoded)
        >>> print(decoded)  # "ATCGATCG"
    """
    
    BASES = ['A', 'C', 'G', 'T']
    BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    @staticmethod
    def one_hot_encode(sequence):
        """
        One-hot encode DNA sequence.
        
        Args:
            sequence (str): DNA sequence (ACGT alphabet)
            
        Returns:
            np.ndarray: One-hot encoded array, shape (4, seq_length)
                       First dimension: [A, C, G, T]
        
        Notes:
            - Input is case-insensitive
            - Ambiguous bases (N, etc.) result in all-zero columns
            - Preserves sequence length exactly
        
        Example:
            >>> encoded = DNAEncoder.one_hot_encode("ACGT")
            >>> print(encoded)
            [[1. 0. 0. 0.]  # A
             [0. 1. 0. 0.]  # C
             [0. 0. 1. 0.]  # G
             [0. 0. 0. 1.]] # T
        """
        seq_upper = sequence.upper()
        encoded = np.zeros((4, len(seq_upper)), dtype=np.float32)
        
        for idx, nucleotide in enumerate(seq_upper):
            if nucleotide in DNAEncoder.BASE_TO_IDX:
                encoded[DNAEncoder.BASE_TO_IDX[nucleotide], idx] = 1.0
        
        return encoded
    
    @staticmethod
    def decode_one_hot(encoded_array, threshold=0.5):
        """
        Decode one-hot encoded array back to DNA sequence.
        
        Args:
            encoded_array (np.ndarray): One-hot array, shape (4, seq_length)
            threshold (float): Activation threshold for base calling
            
        Returns:
            str: Decoded DNA sequence
        
        Notes:
            - Uses argmax for base calling (ignores threshold if any channel > 0)
            - All-zero columns are decoded as 'N'
        
        Example:
            >>> encoded = np.array([[1,0,0,0],
            ...                     [0,1,0,0],
            ...                     [0,0,1,0],
            ...                     [0,0,0,1]], dtype=np.float32)
            >>> DNAEncoder.decode_one_hot(encoded)
            'ACGT'
        """
        sequence = []
        
        for i in range(encoded_array.shape[1]):
            col = encoded_array[:, i]
            
            if np.max(col) < threshold:
                sequence.append('N')  # Ambiguous base
            else:
                base_idx = np.argmax(col)
                sequence.append(DNAEncoder.IDX_TO_BASE[base_idx])
        
        return ''.join(sequence)
    
    @staticmethod
    def ordinal_encode(sequence):
        """
        Simple ordinal encoding: A=0, C=1, G=2, T=3.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            np.ndarray: Ordinal encoded array, shape (seq_length,)
        
        Example:
            >>> DNAEncoder.ordinal_encode("ACGT")
            array([0, 1, 2, 3])
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        return np.array([mapping.get(n.upper(), 0) for n in sequence], dtype=np.int64)
    
    @staticmethod
    def compute_gc_content(sequence):
        """
        Calculate GC content percentage.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            float: GC content as percentage (0-100)
        
        Example:
            >>> DNAEncoder.compute_gc_content("ATCGATCG")
            50.0
        """
        seq_upper = sequence.upper()
        gc_count = seq_upper.count('G') + seq_upper.count('C')
        total = len(seq_upper)
        
        return (gc_count / total) * 100 if total > 0 else 0.0
    
    @staticmethod
    def reverse_complement(sequence):
        """
        Generate reverse complement of DNA sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            str: Reverse complement
        
        Example:
            >>> DNAEncoder.reverse_complement("ATCG")
            'CGAT'
        """
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base.upper(), 'N') for base in reversed(sequence))