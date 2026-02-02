# Preprocessing Method Explanations

This directory contains **individual, focused visualizations** for each of the 11 preprocessing methods used in the FreshRetailNet-50K sales forecasting pipeline.

## 📸 Individual Method Images

Each image provides a clear, interpretable explanation of what that specific preprocessing method does:

### 1. Parse Dates
**File:** `method_01_parse_dates.png`  
**Shows:** Date type conversion from string to datetime object  
**Before:** `'2024-03-28'` (string)  
**After:** `2024-03-28 00:00:00` (datetime64[ns])

### 2. Select Columns
**File:** `method_02_select_columns.png`  
**Shows:** Which columns are kept vs dropped  
**Purpose:** Focus on relevant features, remove sequence data

### 3. Sort by Entity and Date
**File:** `method_03_sort_data.png`  
**Shows:** Chronological sorting visualization  
**Purpose:** Ensure correct order for lag/rolling calculations

### 4. Impute Missing Values
**File:** `method_04_impute_missing.png`  
**Shows:** Missing value counts before/after, imputation statistics  
**Method:** Median (numeric), Mode (categorical)  
**Fit on:** Training data only

### 5. Add Temporal Features
**File:** `method_05_temporal_features.png`  
**Shows:** Distribution of temporal features (month, day of week, quarter)  
**Features Created:** 13 (year, month, week, day_of_week, quarter, sin/cos encodings, etc.)  
**Purpose:** Capture seasonality and time patterns

### 6. Add Lag Features
**File:** `method_06_lag_features.png`  
**Shows:** Correlation plots between target and lag features  
**Features Created:** 4 (lag_1, lag_2, lag_4, lag_7)  
**Purpose:** Capture autoregressive patterns (past predicts future)

### 7. Add Rolling Features
**File:** `method_07_rolling_features.png`  
**Shows:** Time series with rolling mean, std, min/max  
**Features Created:** 12 (mean, std, min, max for windows 4, 7, 14 days)  
**Purpose:** Smooth trends and capture volatility

### 8. Add External Features  
**File:** `method_08_external_features.png`  
**Shows:** Correlation between external features and target  
**Features:** Weather (temp, precipitation), promotions (discount, holiday), stock  
**Features Created:** 16 (base features + their lags)  
**Purpose:** Incorporate external factors affecting sales

### 9. Encode Categoricals
**File:** `method_09_encode_categoricals.png`  
**Shows:** Category encoding transformation  
**Method:** Label encoding (store_id, product_id, category_ids)  
**Fit on:** Training data only (unknown → -1)

### 10. Cap Outliers
**File:** `method_10_cap_outliers.png`  
**Shows:** Box plots before/after outlier capping  
**Method:** Winsorize at 1st and 99th percentiles  
**Fit on:** Training data only

### 11. Scale Numerical
**File:** `method_11_scale_numerical.png`  
**Shows:** Distribution shift from original to standardized  
**Method:** StandardScaler (mean=0, std=1)  
**Fit on:** Training data only

## 🎯 Key Features

Each visualization:
- ✅ **Focused:** Shows only the specific method, not combined plots
- ✅ **Clear:** Easy to understand what changes are made
- ✅ **Interpretable:** Includes statistics, correlations, and examples
- ✅ **Standalone:** Can be used individually in presentations
- ✅ **High Quality:** 150 DPI, suitable for slides and reports

## 📊 Use Cases

### For Presentations
- Use individual images as slides
- Each method gets its own slide
- Clear before/after comparisons

### For Documentation
- Include in reports, papers, or blog posts
- Reference specific methods
- Show concrete examples

### For Teaching
- Explain preprocessing step-by-step
- Visual aids for each concept
- Real data examples

## 🔄 Regenerating Images

To regenerate all images:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the generator script
python generate_preprocessing_explanations.py
```

This will:
1. Load FreshRetailNet-50K dataset (5,000 sample rows)
2. Apply each preprocessing method sequentially
3. Generate 11 individual PNG files
4. Save to `preprocessing_explanations/` directory

**Time:** ~10-15 seconds

## 📏 Image Specifications

- **Format:** PNG
- **DPI:** 150
- **Size:** 14" × 6" to 14" × 10" (varies by method)
- **File Size:** 60KB - 190KB per image
- **Total:** ~1.2 MB for all 11 images

## 🆚 Difference from `preprocessing_analysis/`

| Directory | Purpose | Image Style |
|-----------|---------|-------------|
| **`preprocessing_analysis/`** | Comprehensive analysis with 7+ plots per step | Multi-panel combined visualizations |
| **`preprocessing_explanations/`** (this folder) | Individual method explanations | Single-focus clear visualizations |

**Use `preprocessing_analysis/`** when you want detailed technical analysis with multiple perspectives.

**Use `preprocessing_explanations/`** when you want clean, focused images for presentations or explanations.

## 📚 Related Documentation

- **Detailed Guide:** [`../PREPROCESSING_DETAILED_GUIDE.md`](../PREPROCESSING_DETAILED_GUIDE.md) - Comprehensive explanations
- **Quick Reference:** [`../docs/PREPROCESSING_11_STEPS.md`](../docs/PREPROCESSING_11_STEPS.md) - Summary table
- **Analysis:** [`../preprocessing_analysis/`](../preprocessing_analysis/) - Detailed multi-panel analysis
- **Code:** [`../src/preprocessing.py`](../src/preprocessing.py) - Implementation
- **Generator:** [`../generate_preprocessing_explanations.py`](../generate_preprocessing_explanations.py) - Script to create these images

## 💡 Tips for Using These Images

### In PowerPoint/Keynote
1. Insert as image
2. Use one image per slide
3. Add title and talking points
4. Highlight key insights with text boxes

### In LaTeX
```latex
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{preprocessing_explanations/method_06_lag_features.png}
\caption{Lag features capture autoregressive patterns}
\label{fig:lag_features}
\end{figure}
```

### In Jupyter Notebooks
```python
from IPython.display import Image, display
display(Image('preprocessing_explanations/method_05_temporal_features.png'))
```

### In Markdown/GitHub
```markdown
![Temporal Features](preprocessing_explanations/method_05_temporal_features.png)
```

## 🎓 Learning Path

**Recommended viewing order for understanding the pipeline:**

1. **Start:** Method 1, 2, 3 (foundation)
2. **Core:** Method 4, 5, 6, 7 (feature engineering)
3. **Advanced:** Method 8 (external features)
4. **Finalize:** Method 9, 10, 11 (model preparation)

## ✅ Summary

You now have:
- ✅ 11 individual, focused explanation images
- ✅ Each method clearly visualized
- ✅ Ready for presentations, reports, or teaching
- ✅ High-quality, professional visualizations
- ✅ Easy to regenerate with provided script

---

**Generated:** February 2, 2025  
**Dataset:** FreshRetailNet-50K (5,000 sample rows)  
**Total Images:** 11  
**Total Size:** ~1.2 MB
