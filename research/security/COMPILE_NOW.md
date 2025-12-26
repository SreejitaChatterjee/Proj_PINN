# How to Compile paper_v3_integrated.tex NOW

MiKTeX has been installed successfully, but there are 3 ways to compile:

---

## Option 1: Use Overleaf (RECOMMENDED - Fastest, no restart needed)

### Why Overleaf?
- No system restart required
- Automatic package management
- Real-time PDF preview
- Industry standard for paper submissions
- **Takes 5 minutes total**

### Steps:
1. Go to https://www.overleaf.com (create free account if needed)
2. Click "New Project" → "Upload Project"
3. Create a ZIP file:
   ```bash
   cd research/security
   powershell Compress-Archive -Path paper_v3_integrated.tex,figures -DestinationPath paper_submission.zip
   ```
4. Upload `paper_submission.zip` to Overleaf
5. Click "Recompile" in Overleaf
6. Download PDF

**Time: ~5 minutes**

---

## Option 2: Compile Locally After Restart (MiKTeX installed)

### Why restart needed?
MiKTeX was just installed via winget. Windows needs to refresh PATH environment variable.

### Steps:
1. **Restart your computer** (or log out and back in)
2. Open terminal in `C:\Users\Lenovo\Proj_PINN\research\security`
3. Run:
   ```bash
   pdflatex paper_v3_integrated.tex
   bibtex paper_v3_integrated
   pdflatex paper_v3_integrated.tex
   pdflatex paper_v3_integrated.tex
   ```
4. PDF will be at `paper_v3_integrated.pdf`

**Time: 10 minutes + restart**

---

## Option 3: Compile NOW with Full Path (Advanced)

If you don't want to restart, use the full MiKTeX path:

### Steps:
1. Find MiKTeX installation:
   ```powershell
   Get-ChildItem -Path "C:\Program Files\MiKTeX" -Filter pdflatex.exe -Recurse
   ```

2. Typical locations:
   - `C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe`
   - `C:\Users\Lenovo\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`

3. Compile with full path:
   ```bash
   cd research/security
   "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" paper_v3_integrated.tex
   "C:\Program Files\MiKTeX\miktex\bin\x64\bibtex.exe" paper_v3_integrated
   "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" paper_v3_integrated.tex
   "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" paper_v3_integrated.tex
   ```

**Time: 5-10 minutes if path is correct**

---

## Which Option Should You Choose?

| Option | Time | Pros | Cons |
|--------|------|------|------|
| **Overleaf** | 5 min | No restart, always works, industry standard | Need internet, upload files |
| **After Restart** | 10 min + restart | Local compilation, full control | Need to restart |
| **Full Path** | 5 min | No restart, local | Need to find correct path |

**Recommendation: Use Overleaf (Option 1)** - It's what reviewers expect and takes 5 minutes.

---

## Expected Output

After compilation, you should see:
- `paper_v3_integrated.pdf` - The compiled paper (~14 pages)
- All 6 figures should render correctly
- No compilation errors

---

## Verification Checklist

After you get the PDF:
- [ ] PDF opens correctly
- [ ] Page count is ~14 pages
- [ ] All 6 figures appear:
  - [ ] Figure 1: Performance comparison (bar charts)
  - [ ] Figure 2: Per-fault performance (bars)
  - [ ] Figure 3: PINN architecture (network diagram)
  - [ ] Figure 4: Training comparison (w=0 vs w=20)
  - [ ] Figure 5: ROC & PR curves
  - [ ] Figure 6: Confusion matrix
- [ ] All 4 tables appear (ablation, comparison, per-fault, computational)
- [ ] References are numbered correctly [1] through [28]
- [ ] No "??" for missing citations

---

## Quick Start: Overleaf Right Now

```bash
# Create ZIP for Overleaf
cd C:/Users/Lenovo/Proj_PINN/research/security
powershell -Command "Compress-Archive -Path paper_v3_integrated.tex,figures -DestinationPath paper_submission.zip -Force"

echo "Now:"
echo "1. Go to https://www.overleaf.com"
echo "2. Upload paper_submission.zip"
echo "3. Click Recompile"
echo "4. Download PDF"
echo ""
echo "ZIP file created at: research/security/paper_submission.zip"
```

**Run the above command now, then upload to Overleaf!**

---

## Troubleshooting

### "Missing package" error in MiKTeX
MiKTeX will automatically download packages when needed. Click "Install" when prompted.

### "Font not found" error
First compilation may take longer (2-3 min) as MiKTeX downloads fonts.

### Figures don't appear
Make sure you uploaded the entire `figures/` folder to Overleaf.

### Compilation timeout in Overleaf
Overleaf free tier has 60s limit. Should be fine for this paper (~30s).

---

**BOTTOM LINE: Use Overleaf - takes 5 minutes, no system restart needed!** 🚀
