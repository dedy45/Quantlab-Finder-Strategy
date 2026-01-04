"""
Data Storage - Efficient data storage and retrieval.

Features:
- Parquet storage for fast I/O
- Partitioned storage by symbol/date
- Caching layer for frequently accessed data
- Metadata management
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from .base import DataConfig, DataResult, DataSource, Resolution

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for stored dataset."""
    name: str
    source: str
    symbols: List[str]
    start_date: str
    end_date: str
    resolution: str
    rows: int
    columns: List[str]
    created_at: str
    updated_at: str
    file_path: str
    file_size_mb: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetMetadata':
        return cls(**data)


class DataStorage:
    """
    Efficient data storage manager.
    
    Supports:
    - Parquet format (recommended for large datasets)
    - CSV format (for compatibility)
    - Partitioned storage by symbol
    - Metadata tracking
    
    Examples
    --------
    >>> storage = DataStorage(base_path='data/processed')
    >>> storage.save(df, name='btc_daily', source='binance')
    >>> df = storage.load('btc_daily')
    """
    
    def __init__(
        self, 
        base_path: str = 'data/processed',
        cache_path: str = 'data/cache',
    ):
        self.base_path = Path(base_path)
        self.cache_path = Path(cache_path)
        self.metadata_file = self.base_path / 'metadata.json'
        
        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self._metadata: Dict[str, DatasetMetadata] = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, DatasetMetadata]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {k: DatasetMetadata.from_dict(v) for k, v in data.items()}
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        data = {k: v.to_dict() for k, v in self._metadata.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save(
        self,
        df: pd.DataFrame,
        name: str,
        source: str = 'unknown',
        format: str = 'parquet',
        partition_by: Optional[str] = None,
        compression: str = 'snappy',
    ) -> str:
        """
        Save DataFrame to storage.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        name : str
            Dataset name (used as filename)
        source : str
            Data source identifier
        format : str
            'parquet' or 'csv'
        partition_by : str, optional
            Column to partition by (e.g., 'symbol')
        compression : str
            Compression algorithm for parquet
            
        Returns
        -------
        str
            Path to saved file
        """
        if format == 'parquet':
            file_path = self.base_path / f"{name}.parquet"
            
            if partition_by and partition_by in df.columns:
                # Partitioned storage
                file_path = self.base_path / name
                df.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression=compression,
                    partition_cols=[partition_by],
                    index=False,
                )
            else:
                df.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression=compression,
                    index=False,
                )
        
        elif format == 'csv':
            file_path = self.base_path / f"{name}.csv"
            df.to_csv(file_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Update metadata
        file_size = self._get_file_size(file_path)
        
        metadata = DatasetMetadata(
            name=name,
            source=source,
            symbols=df['symbol'].unique().tolist() if 'symbol' in df.columns else [],
            start_date=str(df['timestamp'].min()) if 'timestamp' in df.columns else '',
            end_date=str(df['timestamp'].max()) if 'timestamp' in df.columns else '',
            resolution='unknown',
            rows=len(df),
            columns=df.columns.tolist(),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            file_path=str(file_path),
            file_size_mb=file_size,
        )
        
        self._metadata[name] = metadata
        self._save_metadata()
        
        logger.info(f"Saved {name}: {len(df)} rows, {file_size:.2f} MB")
        return str(file_path)
    
    def load(
        self,
        name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load DataFrame from storage.
        
        Parameters
        ----------
        name : str
            Dataset name
        columns : List[str], optional
            Columns to load (for efficiency)
        filters : List[tuple], optional
            Parquet filters, e.g., [('symbol', '==', 'BTCUSDT')]
        use_cache : bool
            Whether to use cache
            
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        # Check cache first
        cache_key = f"{name}_{hash(str(columns))}_{hash(str(filters))}"
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Find file
        parquet_path = self.base_path / f"{name}.parquet"
        csv_path = self.base_path / f"{name}.csv"
        dir_path = self.base_path / name  # Partitioned
        
        if parquet_path.exists():
            df = pd.read_parquet(
                parquet_path,
                columns=columns,
                filters=filters,
            )
        elif dir_path.exists() and dir_path.is_dir():
            df = pd.read_parquet(
                dir_path,
                columns=columns,
                filters=filters,
            )
        elif csv_path.exists():
            df = pd.read_csv(csv_path, usecols=columns)
        else:
            raise FileNotFoundError(f"Dataset not found: {name}")
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def load_symbols(
        self,
        name: str,
        symbols: List[str],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load specific symbols from dataset.
        
        Parameters
        ----------
        name : str
            Dataset name
        symbols : List[str]
            Symbols to load
        columns : List[str], optional
            Columns to load
            
        Returns
        -------
        pd.DataFrame
            Filtered data
        """
        filters = [('symbol', 'in', symbols)]
        return self.load(name, columns=columns, filters=filters)
    
    def load_date_range(
        self,
        name: str,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load data for specific date range.
        
        Parameters
        ----------
        name : str
            Dataset name
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        columns : List[str], optional
            Columns to load
            
        Returns
        -------
        pd.DataFrame
            Filtered data
        """
        df = self.load(name, columns=columns)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            df = df[mask]
        
        return df
    
    def list_datasets(self) -> List[DatasetMetadata]:
        """List all available datasets."""
        return list(self._metadata.values())
    
    def get_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Get metadata for specific dataset."""
        return self._metadata.get(name)
    
    def delete(self, name: str) -> bool:
        """Delete a dataset."""
        if name not in self._metadata:
            return False
        
        metadata = self._metadata[name]
        file_path = Path(metadata.file_path)
        
        if file_path.exists():
            if file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
            else:
                file_path.unlink()
        
        del self._metadata[name]
        self._save_metadata()
        
        logger.info(f"Deleted dataset: {name}")
        return True
    
    def _get_file_size(self, path: Path) -> float:
        """Get file size in MB."""
        if path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        else:
            total = path.stat().st_size
        return total / (1024 * 1024)
    
    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Load from cache if exists and fresh."""
        cache_file = self.cache_path / f"{key}.parquet"
        
        if cache_file.exists():
            # Check if cache is fresh (less than 1 hour old)
            age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if age < 3600:  # 1 hour
                return pd.read_parquet(cache_file)
        
        return None
    
    def _save_to_cache(self, key: str, df: pd.DataFrame):
        """Save to cache."""
        cache_file = self.cache_path / f"{key}.parquet"
        df.to_parquet(cache_file, engine='pyarrow', compression='snappy')
    
    def clear_cache(self):
        """Clear all cached data."""
        for f in self.cache_path.glob('*.parquet'):
            f.unlink()
        logger.info("Cache cleared")


# Convenience functions
def save_data(
    df: pd.DataFrame,
    name: str,
    path: str = 'data/processed',
    **kwargs
) -> str:
    """Quick save function."""
    storage = DataStorage(base_path=path)
    return storage.save(df, name, **kwargs)


def load_data(
    name: str,
    path: str = 'data/processed',
    **kwargs
) -> pd.DataFrame:
    """Quick load function."""
    storage = DataStorage(base_path=path)
    return storage.load(name, **kwargs)
