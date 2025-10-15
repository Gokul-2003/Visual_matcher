 Visual Product Matcher

Visual Product Matcher** is a web application that allows users to find visually similar products in a catalog. Upload an image of a product, and the system will return the closest matching items from your dataset.

 🔹 Features

- Upload product images (JPG, JPEG, PNG, WEBP) or provide an image URL.
- Find visually similar products from a predefined dataset.
- Display results with images in an easy-to-view format.
- Handles multiple image formats.
- Works locally and deployed on Streamlit.

 🔹 Folder Structure
 
visual_product_matcher/
│
├─ app.py # Main Streamlit application
├─ requirements.txt # Python dependencies
├─ data/123/ # Folder containing product images
│ ├─ 1.jpg
│ ├─ 2.jpg
│ ├─ 10.jpg
│ └─ ... other images
├─ venv/ # Virtual environment (ignored in .gitignore)
├─ README.md # Project documentation

 **Note:** The `venv` folder is not included in the repository. Only add your data folder and code files.

 🔹 Installation

1. Clone the repository:

git clone https://github.com/your-username/visual_product_matcher.git
cd visual_product_matcher

2. Create and activate a virtual environment:

python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Run the Streamlit app locally:

streamlit run app.py
🔹 Usage

Upload a product image or enter an image URL.

Click Query to find visually similar products.

View the results below the input section.

🔹 Deployment

The app can be deployed directly to Streamlit Cloud

Make sure the data/123/ folder is included in the repo when deploying.

The app handles missing files gracefully, but all dataset images should ideally be present for full functionality.

🔹 Requirements

Python 3.10+

Streamlit

PIL (Pillow)

Other packages as listed in requirements.txt
