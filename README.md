# Evaluating the Impact of Digital Image Processing on Model Performance in Visual Classification

This project evaluates how various Digital Image Processing (DIP) techniques affect the performance of AlexNet on a pest classification task. 

The work is divided into two main stages:
1. **Model Selection** – Comparison and evaluation of different CNN architectures.
2. **DIP Implementation** – Application of image processing techniques to understand their impact on classification performance.

---

### Dataset Source

- **UCI Machine Learning Repository**
- **Dataset Link:** [https://archive.ics.uci.edu/dataset/920/jute+pest+dataset](https://archive.ics.uci.edu/dataset/920/jute+pest+dataset)

---

### Included Files

- `model_selection_part_01.ipynb`: Fine-tuning and evaluation of seven CNN models.
- `model_selection_part_02.ipynb`: Further model comparisons using six additional architectures.
- `DIP_implementation_part_01.ipynb`: DIP techniques like histogram equalization, Gaussian blur, etc.
- `DIP_implementation_part_02.ipynb`: Additional DIP methods including morphological operations and edge detection.

---

### How to Run the Notebooks in Google Colab (from GitHub)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click on the **"GitHub"** tab.
3. Paste the repository link:  
   `https://github.com/fms-faisal/Impact-of-Digital-Image-Processing-on-Model-Performance.git`
4. Press Enter, and Colab will show a list of notebooks from the repo.
5. Click on any notebook (e.g., `model_selection_part_01.ipynb`) to open it in Colab.
6. Use `Shift + Enter` to run each cell.

Alternatively, use this code in a Colab notebook to clone and access the project manually:

```python
!git clone https://github.com/fms-faisal/Impact-of-Digital-Image-Processing-on-Model-Performance.git
%cd Impact-of-Digital-Image-Processing-on-Model-Performance
```
### Note

- You may need to **mount Google Drive** or **upload the dataset manually**.
- Make sure to **update dataset paths** in the code to match your environment.

