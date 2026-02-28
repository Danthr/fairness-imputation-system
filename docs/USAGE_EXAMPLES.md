# Usage Examples

Complete code examples for using the ML Backend API.

---

## Example 1: Basic Workflow (Python)
```python
import requests
import json

BASE_URL = "http://localhost:5000"

# Step 1: Upload dataset
print("Uploading dataset...")
with open('data/raw/test_data.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{BASE_URL}/api/upload", files=files)
    upload_result = response.json()
    
dataset_id = upload_result['dataset_id']
print(f"✓ Uploaded: {upload_result['filename']}")
print(f"  Dataset ID: {dataset_id}")
print(f"  Rows: {upload_result['stats']['total_rows']}")
print(f"  Missing: {upload_result['stats']['missing_percentage']}%")

# Step 2: Process dataset
print("\nProcessing dataset...")
payload = {
    "dataset_id": dataset_id,
    "n_neighbors": 5
}
response = requests.post(f"{BASE_URL}/api/process", json=payload)
process_result = response.json()

print(f"✓ Processing complete")
print(f"  Values imputed: {process_result['imputation']['stats']['values_imputed']}")
print(f"  Time taken: {process_result['imputation']['stats']['execution_time_seconds']}s")

# Step 3: Display fairness results
print("\nFairness Audit Results:")
for attr, result in process_result['fairness_audit'].items():
    assessment = result['overall_assessment']
    print(f"  {attr}: {assessment['summary']}")
    print(f"    Metrics passed: {assessment['metrics_passed']}/{assessment['total_metrics']}")

print(f"\n✓ Output file: {process_result['output_file']}")
```

---

## Example 2: Streamlit Frontend
```python
import streamlit as st
import requests
import pandas as pd

BASE_URL = "http://localhost:5000"

st.title("Fairness Imputation System")

# File upload
uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'])

if uploaded_file:
    # Upload to API
    files = {'file': uploaded_file}
    response = requests.post(f"{BASE_URL}/api/upload", files=files)
    result = response.json()
    
    if response.status_code == 200:
        dataset_id = result['dataset_id']
        st.success(f"✓ File uploaded: {result['filename']}")
        
        # Display stats
        stats = result['stats']
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", stats['total_rows'])
        col2.metric("Total Columns", stats['total_columns'])
        col3.metric("Missing %", f"{stats['missing_percentage']}%")
        
        # Process button
        if st.button("Process Dataset"):
            with st.spinner("Processing..."):
                payload = {"dataset_id": dataset_id}
                response = requests.post(f"{BASE_URL}/api/process", json=payload)
                process_result = response.json()
                
                if response.status_code == 200:
                    st.success("✓ Processing complete!")
                    
                    # Imputation results
                    st.subheader("Imputation Results")
                    imp_stats = process_result['imputation']['stats']
                    st.write(f"Values imputed: {imp_stats['values_imputed']}")
                    st.write(f"Time: {imp_stats['execution_time_seconds']}s")
                    
                    # Fairness results
                    st.subheader("Fairness Audit")
                    for attr, audit in process_result['fairness_audit'].items():
                        with st.expander(f"{attr.upper()}"):
                            assessment = audit['overall_assessment']
                            st.write(f"**Status:** {assessment['summary']}")
                            st.write(f"**Metrics Passed:** {assessment['metrics_passed']}/{assessment['total_metrics']}")
                    
                    # Download link
                    st.download_button(
                        "Download Processed Dataset",
                        data=open(process_result['output_file'], 'rb'),
                        file_name="processed_data.csv"
                    )
```

---

## Example 3: React Frontend (JavaScript)
```javascript
import React, { useState } from 'react';
import axios from 'axios';

const BASE_URL = 'http://localhost:5000';

function App() {
  const [file, setFile] = useState(null);
  const [datasetId, setDatasetId] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  // Upload file
  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post(`${BASE_URL}/api/upload`, formData);
      setDatasetId(response.data.dataset_id);
      alert('File uploaded successfully!');
    } catch (error) {
      alert('Upload failed: ' + error.message);
    }
  };

  // Process dataset
  const handleProcess = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${BASE_URL}/api/process`, {
        dataset_id: datasetId,
        n_neighbors: 5
      });
      setResults(response.data);
      alert('Processing complete!');
    } catch (error) {
      alert('Processing failed: ' + error.message);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Fairness Imputation System</h1>
      
      {/* File Upload */}
      <div>
        <input 
          type="file" 
          onChange={(e) => setFile(e.target.files[0])}
          accept=".csv,.xlsx,.xls"
        />
        <button onClick={handleUpload}>Upload</button>
      </div>

      {/* Process Button */}
      {datasetId && (
        <button onClick={handleProcess} disabled={loading}>
          {loading ? 'Processing...' : 'Process Dataset'}
        </button>
      )}

      {/* Results */}
      {results && (
        <div>
          <h2>Results</h2>
          <p>Values Imputed: {results.imputation.stats.values_imputed}</p>
          <p>Output File: {results.output_file}</p>
          
          <h3>Fairness Audit</h3>
          {Object.entries(results.fairness_audit).map(([attr, audit]) => (
            <div key={attr}>
              <h4>{attr}</h4>
              <p>{audit.overall_assessment.summary}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
```

---

## Example 4: Command Line Script
```python
#!/usr/bin/env python3
"""
Command line tool to process datasets
Usage: python process_dataset.py input.csv output.csv
"""

import sys
import requests

BASE_URL = "http://localhost:5000"

def process_dataset(input_file, output_file):
    # Upload
    with open(input_file, 'rb') as f:
        response = requests.post(f"{BASE_URL}/api/upload", files={'file': f})
        dataset_id = response.json()['dataset_id']
    
    # Process
    payload = {"dataset_id": dataset_id}
    response = requests.post(f"{BASE_URL}/api/process", json=payload)
    result = response.json()
    
    # Copy output file
    import shutil
    shutil.copy(result['output_file'], output_file)
    
    print(f"✓ Processed: {input_file} → {output_file}")
    print(f"  Imputed: {result['imputation']['stats']['values_imputed']} values")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_dataset.py input.csv output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_dataset(input_file, output_file)
```

**Usage:**
```bash
python process_dataset.py my_data.csv clean_data.csv
```

---

## Example 5: Batch Processing Multiple Files
```python
import requests
import os
from pathlib import Path

BASE_URL = "http://localhost:5000"

def process_all_files(input_dir, output_dir):
    """Process all CSV files in a directory"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    for file in os.listdir(input_dir):
        if not file.endswith('.csv'):
            continue
        
        print(f"\nProcessing: {file}")
        input_path = os.path.join(input_dir, file)
        
        # Upload
        with open(input_path, 'rb') as f:
            response = requests.post(f"{BASE_URL}/api/upload", files={'file': f})
            if response.status_code != 200:
                print(f"  ✗ Upload failed")
                continue
            dataset_id = response.json()['dataset_id']
        
        # Process
        payload = {"dataset_id": dataset_id}
        response = requests.post(f"{BASE_URL}/api/process", json=payload)
        if response.status_code != 200:
            print(f"  ✗ Processing failed")
            continue
        
        result = response.json()
        print(f"  ✓ Imputed: {result['imputation']['stats']['values_imputed']} values")
        
        # Copy output
        import shutil
        output_path = os.path.join(output_dir, file)
        shutil.copy(result['output_file'], output_path)
        print(f"  ✓ Saved to: {output_path}")

# Process all files
process_all_files('data/raw', 'data/outputs')
```

---

## Example 6: Error Handling
```python
import requests

BASE_URL = "http://localhost:5000"

def safe_upload(file_path):
    """Upload with proper error handling"""
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/api/upload",
                files={'file': f},
                timeout=30
            )
        
        if response.status_code == 200:
            return response.json()['dataset_id'], None
        else:
            error = response.json().get('error', 'Unknown error')
            return None, f"Upload failed: {error}"
    
    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except requests.ConnectionError:
        return None, "Cannot connect to API. Is it running?"
    except requests.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Usage
dataset_id, error = safe_upload('my_data.csv')
if error:
    print(f"Error: {error}")
else:
    print(f"Success! Dataset ID: {dataset_id}")
```

---

## Tips for Frontend Developers

1. **Always check response status codes**
```python
   if response.status_code == 200:
       # Success
   elif response.status_code == 400:
       # Bad request - show error to user
   elif response.status_code == 500:
       # Server error - try again later
```

2. **Show loading indicators**
   - Processing can take a few seconds
   - Keep users informed

3. **Handle errors gracefully**
   - Display user-friendly error messages
   - Provide helpful suggestions

4. **Validate files before upload**
   - Check file size (< 16MB)
   - Check file type (.csv, .xlsx, .xls)

5. **Save dataset IDs**
   - Store them in state/session
   - Allow users to view past results