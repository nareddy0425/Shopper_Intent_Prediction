# 🛒 E-Commerce Purchase Intent Prediction & Customer Segmentation

## 📌 Project Overview
In e-commerce, treating every website visitor identically results in operational inefficiency and diluted profit margins. This project builds a data-driven framework in R to decode the "digital body language" of online shoppers and predict real-time purchase intent. 

By analyzing session-level browsing behavior (such as page views, exit rates, and administrative dwell time), we construct predictive classification models that identify high-intent buyers and flag sessions at risk of abandonment. Additionally, we leverage unsupervised learning to uncover hidden customer segments, transforming raw clickstream data into clear, actionable business strategies.

---

## 💼 Core Business Implications
Models are only as valuable as the strategic decisions they empower. This analytics framework serves as a scalable asset for digital product and marketing teams to:
* **Optimize Resource Allocation:** Prevent margin erosion by targeting promotional incentives (e.g., real-time 15% discount pop-ups or free shipping thresholds) *only* to on-the-fence shoppers flagged as highly likely to abandon their carts.
* **Proactive Cart Recovery:** Trigger live-chat interventions or targeted retargeting emails the moment a user's behavior mimics a high exit-rate pattern.
* **Tailored User Experiences:** Utilize customer clusters to personalize homepage layouts, product recommendations, and marketing campaigns for distinct shopper personas.

---

## 🛠️ Tech Stack & Libraries
* **Language:** R (v4.3+)
* **Machine Learning & Modeling:** `randomForest`, `caret`, `car`
* **Data Manipulation & Unsupervised Learning:** `dplyr`, `reshape2`, `stats` (K-Means & PCA)
* **Data Visualization:** `ggplot2`, `corrplot`, `RColorBrewer`

---

## 📊 Key Insights & Visualizations

### Feature Correlation
Our exploratory data analysis revealed the interconnected system driving user conversion. Notably, behavioral attributes strongly dictate final outcomes over temporal ones.

![Feature Correlation Heatmap](heatmap.png)

> **Figure 1:** A feature correlation matrix mapping the linear relationships between browsing behaviors and purchase outcomes, highlighting that `PageValues` serves as a dominant positive indicator of buyer intent while elevated `ExitRates` strongly signal session abandonment.

### Core Data Takeaways:
1. **The Power of Page Value:** `PageValues` (the financial worth assigned to a page based on its historical contribution to sales) holds the strongest positive correlation with actual revenue generation. 
2. **Friction Points:** High `ExitRates` and `BounceRates` act as heavy negative anchors. Minimizing friction on high-traffic pages directly mitigates session drop-offs.
3. **Intentional vs. Casual Browsing:** Product-related browsing duration increases conversion velocity far more effectively than administrative page interactions.

---

## 🚀 Modeling Approach & Methodology

The project structure navigates through a comprehensive analytics pipeline:

1. **Exploratory Data Analysis (EDA):** Visualizing features against revenue to isolate behavioral patterns across weekends, visitor types, and seasonal months.
2. **Predictive Modeling Framework:**
   * **Logistic Regression:** Deployed to isolate variables that strictly impact conversion probability, ensuring model parsimony.
   * **Random Forest (Classification):** Utilized an ensemble of 500 decision trees to maximize non-linear predictive accuracy and generate strong classification baselines.
3. **Advanced Feature Engineering (Clustering & PCA):** * Applied **K-Means Clustering** to segment visitors into distinct behavioral buckets based on engagement levels.
   * Leveraged **Principal Component Analysis (PCA)** to reduce feature dimensionality, concentrating variance into orthogonal components.
   * Evaluated a secondary hybrid Random Forest model combining engineered PCA components and cluster assignments to optimize prediction latency and accuracy.

---

## 📂 Repository Structure
```text
├── DataAnalyticsFunctions.R   # Source script containing helper evaluation metrics
├── Shopper_Intent_Prediction.R # Main R script executing EDA, modeling, and clustering
├── online_shoppers_intention.csv # E-commerce clickstream session dataset
└── README.md                  # Project documentation and business write-up

⚙️ How to Run the Project
Clone this repository to your local machine:

Bash
git clone [https://github.com/YOUR_USERNAME/ecommerce-purchase-prediction.git](https://github.com/YOUR_USERNAME/ecommerce-purchase-prediction.git)
Open RStudio and set your working directory to the cloned repository folder.

Ensure all required packages are installed by running:

R
install.packages(c("randomForest", "caret", "ggplot2", "dplyr", "reshape2", "corrplot", "RColorBrewer"))
Execute the main analytical script:

R
source("Shopper_Intent_Prediction.R")
