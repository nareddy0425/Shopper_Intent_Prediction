# Predicting What Makes Window Shoppers Hit Buy
source("DataAnalyticsFunctions.R")


# Libraries
library(randomForest)
library(caret)
library(lattice)
library(ggplot2)
library(dplyr)
library(reshape2)
library(corrplot)
library(RColorBrewer)
library(car)


###################################
#    Functions
###################################

calc_metrics <- function(model, test_data, target_col, threshold = 0.5) {
  # Predict probabilities
  pred_probs <- predict(model, newdata = test_data, type = "response")
  
  # Convert probabilities to binary predictions
  pred_class <- ifelse(pred_probs > threshold, 1, 0)
  
  # Ensure both factors have same levels
  true_labels <- factor(test_data[[target_col]], levels = c(0, 1))
  pred_class <- factor(pred_class, levels = c(0, 1))
  
  # Confusion matrix
  cm <- confusionMatrix(pred_class, true_labels)
  
  TP <- cm$table[2, 2]
  TN <- cm$table[1, 1]
  FP <- cm$table[1, 2]
  FN <- cm$table[2, 1]
  
  # Metrics
  accuracy  <- (TP + TN) / (TP + TN + FP + FN)
  TPR       <- TP / (TP + FN)   # Sensitivity / Recall
  TNR       <- TN / (TN + FP)   # Specificity
  Precision <- TP / (TP + FP)
  F1        <- 2 * (Precision * TPR) / (Precision + TPR)
  
  # Return as list
  return(list(
    Accuracy = accuracy,
    Precision = Precision,
    Recall = TPR,
    F1_Score = F1
  ))
}

calc_rf_metrics <- function(model, test_data, target_col, threshold = 0.5) {
  # Extract probabilities for class "1"
  pred_probs <- predict(model, test_data, type = "prob")[, "1"]
  
  # Convert probabilities to binary predictions based on threshold
  pred_class <- ifelse(pred_probs > threshold, 1, 0)
  
  # Ensure both factors have same levels
  true_labels <- factor(test_data[[target_col]], levels = c(0, 1))
  pred_class <- factor(pred_class, levels = c(0, 1))
  
  # Confusion matrix
  cm <- caret::confusionMatrix(pred_class, true_labels)
  
  TP <- cm$table[2, 2]
  TN <- cm$table[1, 1]
  FP <- cm$table[1, 2]
  FN <- cm$table[2, 1]
  
  # Metrics
  accuracy  <- (TP + TN) / (TP + TN + FP + FN)
  precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  recall    <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  f1_score  <- ifelse(is.na(precision) | is.na(recall) | (precision + recall) == 0,
                      NA,
                      2 * (precision * recall) / (precision + recall))
  
  # Return as list
  return(list(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  ))
}


###################################
#    EXPLORATORY DATA ANALYSIS
###################################

df1 <- read.csv("online_shoppers_intention.csv")

# Structure
str(df1)
head(df1)

# We see that there are no null values

# Categorical variables
month_mapping <- c("Jan"=1, "Feb"=2, "Mar"=3, "Apr"=4, "May"=5, 
                   "June"=6, "Jul"=7, "Aug"=8, "Sep"=9, "Oct"=10, "Nov"=11, "Dec"=12)
df1$Month <- month_mapping[df1$Month]

visitor_mapping <- c("New_Visitor"=0, "Returning_Visitor"=1, "Other"=2)
df1$VisitorType <- visitor_mapping[df1$VisitorType]

# Revenue distribution
ggplot(df1, aes(x = Revenue)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Revenue Distribution") +
  xlab("Revenue") + ylab("Count")

# Select only numeric columns
numeric_df <- df1 %>% select_if(is.numeric)

# Compute correlation matrix
corr_matrix <- cor(numeric_df, use = "complete.obs")

# Melt for ggplot
corr_melt <- melt(corr_matrix)

# Plot heatmap
ggplot(data = corr_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name = "Correlation") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.text.y = element_text(size = 10),
        panel.grid = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  coord_fixed() +
  ggtitle("Correlation Heatmap")

# Conversion rate by Visitor Type
visitor_summary <- df1 %>%
  group_by(VisitorType) %>%
  summarise(ConversionRate = mean(Revenue))

ggplot(visitor_summary, aes(x = factor(VisitorType), y = ConversionRate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  ggtitle("Conversion Rate by Visitor Type") +
  xlab("Visitor Type") + ylab("Conversion Rate")

# Conversion rate by Month
month_summary <- df1 %>%
  group_by(Month) %>%
  summarise(ConversionRate = mean(Revenue))

ggplot(month_summary, aes(x = Month, y = ConversionRate)) +
  geom_line(group = 1, color = "darkorange", linewidth = 1) +
  geom_point(color = "darkorange", size = 2) +
  ggtitle("Conversion Rate by Month") +
  xlab("Month") + ylab("Conversion Rate")

# Weekend vs Weekday conversion
weekend_summary <- df1 %>%
  group_by(Weekend) %>%
  summarise(ConversionRate = mean(Revenue))

ggplot(weekend_summary, aes(x = Weekend, y = ConversionRate, fill = Weekend)) +
  geom_bar(stat = "identity") +
  ggtitle("Conversion Rate: Weekend vs Weekday") +
  xlab("Weekend") + ylab("Conversion Rate") +
  theme(legend.position = "none")

# Boxplot: PageValues vs Revenue
ggplot(df1, aes(x = as.factor(Revenue), y = PageValues, fill = as.factor(Revenue))) +
  geom_boxplot(outlier.color = "black") +
  ggtitle("Page Values vs Revenue") +
  xlab("Revenue") + ylab("Page Values") +
  theme(legend.position = "none")

# Boxplot: BounceRates vs Revenue
ggplot(df1, aes(x = as.factor(Revenue), y = BounceRates, fill = as.factor(Revenue))) +
  geom_boxplot(outlier.color = "black") +
  ggtitle("Bounce Rates vs Revenue") +
  xlab("Revenue") + ylab("Bounce Rate") +
  theme(legend.position = "none")

# Scatter: BounceRates vs ExitRates
ggplot(df1, aes(x = BounceRates, y = ExitRates)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  ggtitle("Bounce Rates vs Exit Rates") +
  xlab("Bounce Rate") + ylab("Exit Rate")

###################################
#    CLEANING/ SPLIT DATAFRAME
###################################

df <- read.csv("DSProject.csv")

# Convert character columns 'Weekend' and 'Revenue' to numeric 0/1
df$Weekend <- ifelse(df$Weekend == "True", 1, 0)
df$Revenue <- ifelse(df$Revenue == "True", 1, 0)

# Set seed for reproducibility
set.seed(143)

n <- nrow(df)
train_index <- sample(1:n, size = 0.8 * n)

# Create train and test dataframes
train <- df[train_index, ]
test  <- df[-train_index, ]

# Convert categorical variables to dummy variables (one-hot encoding)
x_train <- model.matrix(~ ., data = train)[, -1]
x_test  <- model.matrix(~ ., data = test)[, -1]

######################################
#   Logistic Regression Model
######################################


base_model <- glm(Revenue ~ ., data = train, family = binomial)

# Show model summary
summary(base_model)

calc_metrics(base_model, test, "Revenue")

# To simplify the logistic regression model, we removed predictors with p-values greater than 0.05, as they do not have a statistically significant effect on Revenue. 
# We kept only Informational, ProductRelated, ExitRates, PageValues, Month, VisitorType, and Weekend, which all showed strong evidence of influence. 
# This makes the model more interpretable and avoids including variables that add noise without improving prediction.

model2 <- glm(Revenue ~ Informational + ProductRelated + ExitRates + PageValues + 
                Month + VisitorType + Weekend,
              data = train,
              family = binomial
)

# View summary
summary(model2)

calc_metrics(model2, test, "Revenue")

vif_values <- vif(model2)

# Convert to a data frame for plotting
vif_df <- data.frame(
  Variable = names(vif_values),
  VIF = vif_values)

# Bar plot
ggplot(vif_df, aes(x = reorder(Variable, -VIF), y = VIF)) +  # descending order
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_hline(yintercept = 4, color = "red", linetype = "dashed") +  # threshold line
  labs(title = "VIF Values for Logistic Regression",
       x = "Variable",
       y = "VIF") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # rotate labels

# There is no multicorrelation. So model2 is our best logistic regression model.

best_log_model <- model2


######################################
#          Random Forest
######################################

# Random Forest
rf_model <- randomForest(as.factor(Revenue) ~ Informational + ProductRelated + ExitRates + PageValues + 
                           Month + VisitorType + Weekend,data = train,
                         ntree = 500,         # number of trees
                         mtry = 3,            # number of variables tried at each split (sqrt of predictors is default)
                         importance = TRUE)   # calculate variable importance

# Print model summary
print(rf_model)



calc_metrics(rf_model, test, "Revenue")
calc_metrics(model2, test, "Revenue")




######################################
#            CLUSTER
######################################


all_scaled <- scale(x_train)

# 4 clusters
k <- 4  
km_res <- kmeans(all_scaled, centers = k, nstart = 20)

train$cluster <- factor(km_res$cluster)


# shows how many observations are in each cluster
print(km_res$size)
# In-Sample R^2 of 0.26 (for explaining deviance of x)
print(1 - sum(km_res$tot.withinss) / km_res$totss)

# variance explained by each PC
pca <- prcomp(all_scaled)
# Sort by abs(PC1) first, then abs(PC2)
rotation_sorted <- pca$rotation[order(-abs(pca$rotation[,1]), -abs(pca$rotation[,2])), 1:2]
# Show top 20
head(rotation_sorted, 20)


# run principal analysis component on the scaled features 
pca <- prcomp(all_scaled)
# plot the clusters based on two principal components
plot(pca$x[,1:2], col = train$cluster,
     pch = 19, cex = 0.6,
     xlab = "PC1", ylab = "PC2",
     main = paste("K-Means Clustering With All Variables"))

# plot clusters based on average income and percentage of people with a bachelors degree 
plot(train$ExitRates, 
     train$ProductRelated_Duration,
     col = train$cluster, 
     pch = 19, cex = 0.6,
     xlab = "Exit Rates",
     ylab = "Product Related Duration",
     main = "Clusters by Exit Rates vs Product Related Duration")



######################################
#               PCA
######################################

# plot PCA
plot(pca,main = "PCA: Variance Explained by Factors", font = 2)
mtext(side=1, "Factors",  line=1, font=2)


## Interpreting the four factors
loadings <- pca$rotation[, 1:4]


# PC1
pc1_loadings <- pca$rotation[, 1]

loadings_df1 <- data.frame(
  Variable = rownames(pca$rotation),
  PC1 = pc1_loadings
)
# Sort by absolute loading
loadings_df1 <- loadings_df1[order(-abs(loadings_df1$PC1)), ]
# View
head(loadings_df1)

# PC2
pc2_loadings <- pca$rotation[, 2]

loadings_df2 <- data.frame(
  Variable = rownames(pca$rotation),
  PC2 = pc2_loadings
)

# Sort by absolute loading
loadings_df2 <- loadings_df2[order(-abs(loadings_df2$PC2)), ]
# View
head(loadings_df2)


# Apply the same scaling and PCA transformation to test data
all_scaled_test <- scale(x_test, center = attr(all_scaled, "scaled:center"), scale = attr(all_scaled, "scaled:scale"))

# Predict cluster assignments for test data based on nearest centroids
test_clusters <- apply(all_scaled_test, 1, function(row) {
  dists <- apply(km_res$centers, 1, function(center) sum((row - center)^2))
  return(which.min(dists))
})

# Add cluster column to both train and test sets
train$cluster <- factor(km_res$cluster)
test$cluster <- factor(test_clusters)

# Get PCA scores for training and test data
train_pca <- as.data.frame(pca$x[, 1:4])  # top 4 PCs
colnames(train_pca) <- paste0("PC", 1:4)

test_pca <- as.data.frame(predict(pca, newdata = all_scaled_test)[, 1:4])
colnames(test_pca) <- paste0("PC", 1:4)

# Combine PCA components, cluster, and target variable
train_rf <- cbind(train_pca, cluster = train$cluster, Revenue = as.factor(train$Revenue))
test_rf  <- cbind(test_pca, cluster = test$cluster, Revenue = as.factor(test$Revenue))


# ---Train Random Forest model ---

rf_model2 <- randomForest(Revenue ~ ., 
                          data = train_rf, 
                          ntree = 500, 
                          mtry = floor(sqrt(ncol(train_rf) - 1)), 
                          importance = TRUE)




# Predictions on test data
rf_pred <- predict(rf_model2, newdata = test_rf)

# Ensure both are factors with same levels
rf_pred <- factor(rf_pred, levels = c(0, 1))
true_labels <- factor(test_rf$Revenue, levels = c(0, 1))

calc_rf_metrics(rf_model2, test_rf, "Revenue", threshold = 0.5)

