import openpolicedata as opd
import pandas as pd
import numpy as np
import requests
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

def get_opd_catalog():
    """Retrieve full OPD catalog with metadata"""
    print("📊 Loading OpenPoliceData catalog...")
    
    # Get the full source table (all agencies/datasets)
    catalog = opd.datasets.query()
    
    print(f"✅ Found {len(catalog)} total datasets in OPD")
    return catalog

def filter_relevant_datasets(catalog):
    """Filter for INCIDENTS, CALLS FOR SERVICE, etc."""
    
    target_types = [
        'INCIDENTS',
        'CALLS FOR SERVICE', 
        # 'USE OF FORCE',
        # 'COMPLAINTS',
        # 'OFFICER-INVOLVED SHOOTINGS',
        # 'TRAFFIC STOPS'
    ]
    
    # OPD uses 'TableType' column
    relevant = catalog[
        (catalog['TableType'].str.upper().isin([t.upper() for t in target_types]))
        & (catalog['Year']=='MULTIPLE')
        & (catalog['DataType']=='Socrata')
    ].copy().reset_index(drop=True)
    
    print(f"\n🎯 Filtered to {len(relevant)} datasets matching target types:")
    print(relevant['TableType'].value_counts())
    
    return relevant

def analyze_dataset_completeness(catalog_subset, max_datasets=None):
    """
    Robust version that works with modern OpenPoliceData (2024–2025+)
    Handles required year/date input, multiple agencies, and edge cases.
    """
    results = []
    
    if max_datasets:
        catalog_subset = catalog_subset.head(max_datasets)
    
    print(f"\nAnalyzing {len(catalog_subset)} datasets for completeness...\n")
    
    for idx, row in catalog_subset.iterrows():
        source_name = row.get('SourceName', 'Unknown')
        state = row.get('State', 'Unknown')
        table_type = row.get('TableType', 'Unknown')
        agency = row.get('Agency', None)
        
        print(f"[{idx+1}/{len(catalog_subset)}] Checking: {source_name} ({state}) - {table_type}...", end="")
        
        try:
            # Step 1: Initialize source with fallback for ambiguous names
            try:
                src = opd.Source(source_name, state=state if state != 'Unknown' else None)
            except Exception as e:
                if "multiple sources" in str(e).lower():
                    print(" (multiple agencies → trying 'MULTIPLE')", end="")
                    src = opd.Source(source_name, state=state, agency="MULTIPLE")
                elif "not found" in str(e).lower() and agency:
                    src = opd.Source(source_name, state=state, agency=agency)
                else:
                    raise e

            # Step 2: Try to load a sample using multiple strategies
            df_sample = None
            loaded_year = None
            
            # Strategy 1: Try recent years first (most likely to exist)
            for year in range(2024, 2014, -1):
                try:
                    df = src.load(table_type=table_type, year=year, nrows=1000)
                    if len(df) > 0:
                        df_sample = df
                        loaded_year = year
                        print(f" ✓ {year}: {len(df)} rows", end="")
                        break
                except:
                    continue
            
            # Strategy 2: If no year worked, try latest available (some sources support this)
            if df_sample is None:
                try:
                    df = src.load(table_type=table_type, nrows=1000)  # no year
                    if len(df) > 0:
                        df_sample = df
                        print(f" ✓ (no filter): {len(df)} rows", end="")
                except:
                    pass
            
            # Strategy 3: Last resort — try a date range
            if df_sample is None:
                try:
                    df = src.load(
                        table_type=table_type,
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                        nrows=1000
                    )
                    if len(df) > 0:
                        df_sample = df
                        loaded_year = 2023
                        print(f" ✓ (range 2023): {len(df)} rows", end="")
                except:
                    pass

            if df_sample is None or len(df_sample) == 0:
                print(" ⚠️ No data loaded")
                results.append({
                    'source_name': source_name,
                    'state': state,
                    'agency': agency or 'Unknown',
                    'table_type': table_type,
                    'status': 'load_failed',
                    'error': 'No data returned for any year/range'
                })
                continue

            print(f" | {len(df_sample.columns)} cols")

            # === Now analyze the loaded sample ===
            date_cols = [col for col in df_sample.columns
                        if any(term in col.lower() for term in ['date', 'time', 'datetime', 'occurred', 'reported', 'incident', 'call', 'stop'])]
            
            text_cols = [col for col in df_sample.columns
                        if any(term in col.lower() for term in ['desc', 'narrative', 'comment', 'notes', 'summary', 'text', 'detail', 'call_type', 'offense', 'incident_type', 'reason', 'violation'])]

            # Parse dates
            date_range_start = date_range_end = None
            years_covered = None
            if date_cols:
                for col in date_cols:
                    try:
                        parsed = pd.to_datetime(
                                df_sample[col],
                                format="mixed",      # This is the magic in pandas ≥ 2.0
                                dayfirst=False,      # set True only if you see dd/mm/yyyy somewhere
                                errors='coerce'
                            )
                        if parsed.notna().any():
                            date_range_start = parsed.min()
                            date_range_end = parsed.max()
                            if date_range_start and date_range_end:
                                years_covered = (date_range_end - date_range_start).days / 365.25
                            break
                    except:
                        continue

            # Field completeness
            field_completeness = {
                col: round((1 - df_sample[col].isna().mean()) * 100, 2)
                for col in df_sample.columns
            }

            # Text field quality
            text_quality = {}
            for col in text_cols:
                if col in df_sample.columns:
                    vals = df_sample[col].dropna().astype(str)
                    if len(vals) > 0:
                        clean = vals.str.strip()
                        non_empty = (clean != '') & (clean != 'nan')
                        text_quality[col] = {
                            'avg_length': round(clean.str.len().mean(), 1),
                            'non_empty_pct': round(non_empty.mean() * 100, 1)
                        }

            results.append({
                'source_name': source_name,
                'state': state,
                'agency': agency or 'Unknown',
                'table_type': table_type,
                'loaded_year': loaded_year,
                'sample_rows': len(df_sample),
                'total_fields': len(df_sample.columns),
                'date_cols': ', '.join(date_cols) if date_cols else None,
                'num_date_cols': len(date_cols),
                'text_cols': ', '.join(text_cols) if text_cols else None,
                'num_text_cols': len(text_cols),
                'date_range_start': date_range_start.date() if date_range_start else None,
                'date_range_end': date_range_end.date() if date_range_end else None,
                'years_covered': round(years_covered, 1) if years_covered else None,
                'avg_field_completeness': round(np.mean(list(field_completeness.values())), 1),
                'text_quality_summary': text_quality or None,
                'status': 'success'
            })

        except Exception as e:
            error_msg = str(e)[:200]
            print(f" ❌ {error_msg}")
            results.append({
                'source_name': source_name,
                'state': state,
                'agency': agency or 'Unknown',
                'table_type': table_type,
                'status': 'source_failed',
                'error': error_msg
            })

    return pd.DataFrame(results)

def download_socrata_full_safe(row, chunk_size=200000, sleep=0.25, max_retries=3):
    source_name = row['SourceName']
    state = row['State']
    table_type = row['TableType'].replace(' ', '_')
    start_year = pd.to_datetime(
            row["coverage_start"],
            format="mixed",      # This is the magic in pandas ≥ 2.0
            dayfirst=False,      # set True only if you see dd/mm/yyyy somewhere
            errors='coerce'
        ).year
    end_year = pd.to_datetime(
            row["coverage_end"],
            format="mixed",      # This is the magic in pandas ≥ 2.0
            dayfirst=False,      # set True only if you see dd/mm/yyyy somewhere
            errors='coerce'
        ).year
    coverage = f"{start_year}-{end_year}"

    # === 1. Get dataset_id safely ===
    dataset_id = row.get('dataset_id')
    if not dataset_id or pd.isna(dataset_id):
        url_field = str(row.get('URL', ''))
        if 'resource/' in url_field:
            dataset_id = url_field.split('resource/')[1].split('.')[0].split('/')[0]
        else:
            print(f"Skipping {source_name}: no dataset_id found")
            return None

    # === 2. Build clean domain ===
    domain = str(row['URL']).strip().lower()
    if domain.startswith('http'):
        domain = domain.split('://')[1]
    domain = domain.split('/')[0]  # remove paths
    domain = domain.rstrip('/')
    
    url = f"https://{domain}/resource/{dataset_id}.json"

    # === 3. Filename ===
    filename = f"{source_name}_{state}_{table_type}_{coverage}.parquet"
    filepath = Path(filename)
    if filepath.exists():
        print(f"   Already exists → {filename} ({filepath.stat().st_size / 1e6:.1f} MB)")
        return pd.read_parquet(filepath)

    print(f"Downloading → {source_name} ({state}) – {row['TableType']}")
    print(f"Coverage: {row['coverage_start'].date()} → {row['coverage_end'].date()} | {dataset_id}")

    all_data = []
    offset = 0
    total = 0
    while True:
        for attempt in range(max_retries):
            try:
                params = {
                    "$limit": chunk_size,
                    "$offset": offset,
                    "$order": row.get('date_field') or "creation_datetime"
                }
                resp = requests.get(url, params=params, timeout=180)
                resp.raise_for_status()
                batch = resp.json()
                
                if not batch:
                    print("No more data. Finished!")
                    break
                    
                df_batch = pd.DataFrame(batch)
                all_data.append(df_batch)
                total += len(df_batch)
                print(f"   +{len(df_batch):,} rows (total: {total:,})")
                offset += chunk_size
                time.sleep(sleep)
                break  # success → exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error (attempt {attempt+1}): {e} → retrying...")
                    time.sleep(10)
                else:
                    print(f"   FAILED after {max_retries} attempts: {e}")
                    return None
        else:
            break  # no break from for → failed
        if not batch:
            break

    if not all_data:
        print("No data downloaded")
        return None

    df = pd.concat(all_data, ignore_index=True)
    # deduplication
    df_temp = df.copy()
    for col in df_temp.columns:
        if df_temp[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df_temp[col] = df_temp[col].astype(str)
    df_temp.drop_duplicates(inplace=True)
    df = df.loc[df_temp.index]  # align back
    df.to_parquet(filepath, index=False)
    size_mb = filepath.stat().st_size / 1e6
    print(f"SAVED → {filename} ({size_mb:.1f} MB) | {len(df):,} rows\n")
    return df

def summarize_downloads(
    directory: str | Path = ".",
    pattern: str = "*.parquet",
    save_csv: bool = True,
    csv_name: str = "download_summary.csv"
) -> pd.DataFrame:
    """
    Scans a directory for Parquet files created by your download_socrata_full_safe()
    and returns a nice summary DataFrame with row counts and sizes.
    """
    directory = Path(directory)
    summary_rows = []

    # Get all matching files first so we can show total count
    files = sorted(directory.glob(pattern))
    total_files = len(files)

    if total_files == 0:
        print("No Parquet files found.")
        return pd.DataFrame()

    print(f"\nScanning {total_files} Parquet file(s)...\n")

    # Counter loop
    for i, file_path in enumerate(files, start=1):
        print(f"  [{i:>{len(str(total_files))}}/{total_files}] Processing {file_path.name}", end="")

        try:
            # Fast metadata read (works with pyarrow/fastparquet)
            df_meta = pd.read_parquet(file_path)  # no columns = just metadata
            n_rows = len(df_meta)

            size_mb = file_path.stat().st_size / 1e6

            # Parse filename
            stem = file_path.stem
            parts = stem.split("_")
            if len(parts) >= 4:
                source_name = parts[0].replace("-", " ")
                state = parts[1]
                table_type = parts[2].replace("_", " ")
                coverage = parts[-1]
            else:
                source_name = stem.split("_")[0]
                state = table_type = coverage = ""

            summary_rows.append({
                "filename": file_path.name,
                "source_name": source_name,
                "state": state,
                "table_type": table_type,
                "coverage": coverage.replace(".parquet", ""),
                "rows": n_rows,
                "size_mb": round(size_mb, 1)
            })
            print(f" → {n_rows:,} rows")  # success indicator

        except Exception as e:
            print(f" → FAILED ({e})")
            print(f"Warning: Could not read {file_path.name}: {e}")

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        print("\nNo valid Parquet files were processed.")
        return summary_df

    # Sort
    summary_df = summary_df.sort_values(
        ["state", "source_name", "table_type", "coverage"]
    ).reset_index(drop=True)

    # Final summary
    print("\n" + "="*90)
    print("DOWNLOAD SUMMARY")
    print("="*90)

    print(f"\nTotal datasets : {len(summary_df)}")
    print(f"Total rows     : {summary_df['rows'].sum():,}")
    print(f"Total size     : {summary_df['size_mb'].sum():,.1f} MB")

    if save_csv:
        out_path = directory / csv_name
        summary_df.to_csv(out_path, index=False)
        print(f"\nSummary saved → {out_path}")

    return summary_df

def load_crime_data_for_nlp(
    directory: str | Path = ".",
    pattern: str = "*.parquet",
    output_path: str = "./data/opd_nlp_dataset.parquet"
) -> pd.DataFrame:
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    total_files = len(files)
    if total_files == 0:
        print("No Parquet files found.")
        return pd.DataFrame()

    desc_cols = [
        'close_type', 'offense_sub_category', 'nibrs_offense_code_description',
        'final_case_type_description', 'call_type', 'incident_type',
        'class_description', 'crime_type', 'incident_type_primary',
        'parent_incident_type', 'type', 'crime', 'primary_type',
        'description', 'incident_type_desc', 'offense', 'agency_crimetype_id',
        'category', 'crimecodedescription', 'nciccodedescription',
        'incidenttype', 'nature', 'incidenttypedescription',
        'call_type_text', 'crm_cd_desc', 'event_type_description',
        'ncodedesc', 'call_type_final_d', 'call_type_final_desc',
        'secondaryeventtype', 'disposition', 'offensecode', 'offense_grouping',
    ]
    date_cols = [
        'creation_datetime', 'createddateutc',
        'received_datetime',
        'occur_begin_date','occur_end_date','report_date',
        'occ_date','rep_date',
        'call_date',
        'response_datetime','call_closed_datetime',
        'incident_datetime',
        'datetimereceived','datetimeclosed',
        'crime_date_time','date_of_report',
        'date',
        'offense_date',
        'date_occ', 'date_rptd',
        'date_reported','date_from',
        'create_time_incident',
        'occurredfromdate','reporteddate',
        'date_occu','date_fnd','date_rept',
        'actdate',
        'incidentstarteddatetime','insertedtimestamp','incidentdate',
        'calltime',
        'dispatch_date',
        'received_date_time', 'dispatch_date_time',
        'call_entry_date',
        'date_time',
        'cad_event_original_time_queued', 'cad_event_arrived_time',
        'start_time',
    ]

    print(f"\nScanning {total_files} Parquet file(s) for NLP-ready OPD descriptions...\n")
   
    all_dfs = []
    for i, file_path in enumerate(files, start=1):
        print(f"[{i:>{len(str(total_files))}}/{total_files}] {file_path.name}", end="")
        try:
            df = pd.read_parquet(file_path)
            # deduplication
            df_temp = df.copy()
            for col in df_temp.columns:
                if df_temp[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df_temp[col] = df_temp[col].astype(str)
            df_temp.drop_duplicates(inplace=True)
            df = df.loc[df_temp.index]  # align back
           
            avail_desc = [c for c in desc_cols if c in df.columns]
            avail_date = [c for c in date_cols if c in df.columns]

            # ────────────────────── OPTIMIZED WITH LIST COMPREHENSION ──────────────────────
            if len(avail_desc) > 1:
                # Convert to list of lists - avoids numpy dtype issues
                text_matrix = df[avail_desc].fillna('').astype(str).replace('nan', '').values.tolist()
                
                # List comprehension with inline strip - fastest approach for object arrays
                crime_desc = [
                    ' | '.join(filter(None, [s.strip() for s in row])) or 'unknown'
                    for row in text_matrix
                ]
                
                df['crime_description'] = crime_desc
                
            elif len(avail_desc) == 1:
                df['crime_description'] = df[avail_desc[0]].fillna('unknown').astype(str)
            else:
                df['crime_description'] = 'unknown'
            # ─────────────────────────────────────────────────────────────────────────────────────

            # Parse dates - use first non-null date per row
            if avail_date:
                date_series_list = [pd.to_datetime(
                                        df[col],
                                        format="mixed",      # This is the magic in pandas ≥ 2.0
                                        dayfirst=False,      # set True only if you see dd/mm/yyyy somewhere
                                        errors='coerce'
                                    )
                                    for col in avail_date]
                incident_date = date_series_list[0]
                for series in date_series_list[1:]:
                    incident_date = incident_date.fillna(series)
            else:
                incident_date = pd.NaT

            # Extract city & state
            stem = file_path.stem
            parts = stem.split("_")
            city_name = parts[0].replace("-", " ")
            state = parts[1] if len(parts) > 1 else "Unknown"

            # Determine table_type using vectorized string operations
            filename_upper = file_path.name.upper()
            if 'INCIDENTS' in filename_upper:
                table_type = 'INCIDENTS'
            elif 'CALLS_FOR_SERVICE' in filename_upper:
                table_type = 'CALLS FOR SERVICE'
            else:
                table_type = 'OTHER'

            # Build DataFrame directly with arrays (faster than dict construction)
            n_rows = len(df)
            df_sel = pd.DataFrame({
                'state': np.full(n_rows, state, dtype=object),
                'city': np.full(n_rows, city_name, dtype=object),
                'incident_date': incident_date,
                'crime_description': df['crime_description'].values,
                'source_file': np.full(n_rows, file_path.name, dtype=object),
                'table_type': np.full(n_rows, table_type, dtype=object)
            })
            df_sel.dropna(inplace=True)
            
            all_dfs.append(df_sel)

            print(f" → {len(df_sel):,} rows | desc: {len(avail_desc)} cols checked | date: {len(avail_date)} cols checked")

        except Exception as e:
            print(f" → FAILED ({e})")

    if not all_dfs:
        print("\nNo data loaded from any file.")
        return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)
    print("\n" + "="*100)
    print("CRIME DATA LOADED SUCCESSFULLY")
    print("="*100)
    print(f"Total files processed : {len(files)}")
    print(f"Total rows combined   : {len(final_df):,}")
    print(f"Final columns         : {list(final_df.columns)}")
    print("="*100)

    print(f"\nSaving final dataset to {output_path}...")
    final_df.to_parquet(
        output_path,
        compression="zstd",      # fastest + best compression
        engine="pyarrow",        # default and most reliable
        index=False
    )

    return final_df