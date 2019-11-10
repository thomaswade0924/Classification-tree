##################
# Main functions #
##################

# tree.grow
#
# @param x numeric dataframe without missing values containing the data to grow the tree on
# @param y vector with equal length as x containg 0, 1 values indicating the class of the rows in x
# @param nmin int minimum number of observations for a node to be allowed to split
# @param minleaf int minimum number of observations for a leaf node. Only splits that obey this value will be considered
# @param nfeat int number of features to be considered at each split
# @return grown tree object, represented as a dataframe
#
# Grows a classification tree to classify the given data.
# nmin, minleaf and nfeat can be used to restrict the growing process and prevent overfitting
tree.grow <- function(x, y, nmin, minleaf, nfeat) {
  # Initializa tree
  tree <- data.frame(feat=character(), split=integer(), mclass=integer(), index_left=integer(), index_right=integer(), stringsAsFactors=FALSE)
  
  # Function used to recursively grow the tree
  #
  # @param x numeric dataframe of observations in this subtree
  # @param y classification of current observations
  # @param index current index in the tree for this subtree
  # @return index
  tree.grow.recursive <- function(x, y, index) {
    # Set majority class on current node
    tree[index, "mclass"] <<- getmode(y)
    
    # Leaf node when the node is pure, or when the size of the node is less than nmin
    if(all(y) | all(!y) | nrow(x) < nmin) {
      return(index)
    }
    
    # Select allowed features for this split
    feats <- sample(names(x))
    
    # Find the best possible split by trying to split at each feature
    feat.best <- NA
    feat.best.split <- NA
    for (feat in feats) {
      # Find best split on this feature
      feat.split <- split.best(x[, feat], y, minleaf)
      
      # Check whether current split is better
      if (length(feat.split$split) > 0 && (is.na(feat.best) || feat.split$quality > feat.best.split$quality)) {
        feat.best <- feat
        feat.best.split <- feat.split
      }
    }
    
    # If no acceptable split can be found, this will be a leaf node
    if (is.na(feat.best)) {
      return(index)
    }
    
    # Fill out this node, and recurse to left and right split
    tree[index, "feat"] <<- feat.best
    tree[index, "split"] <<- feat.best.split$split
    l <- x[[feat.best]] <= feat.best.split$split
    tree[index, "index_left"] <<- tree.grow.recursive(x[l,], y[l], index+1)
    r <- x[[feat.best]] > feat.best.split$split
    tree[index, "index_right"] <<- tree.grow.recursive(x[r,], y[r], nrow(tree)+1)
    
    return(index)
  }
  
  # Grow tree recursively from the root
  tree.grow.recursive(x, y, 1)
  
  return(tree)
}

# tree.classify
#
# @param x numeric dataframe with similar features as the tree. These rows will be classified
# @param tr tree object used to classify the rows
# @return vector with the classification for each row of x
#
# Classifies new rows using a classification tree
tree.classify <- function(x, tr) {
  y <- vector()
  
  # Classify each row from x
  for(i in 1:nrow(x)){
    index <- 1
    
    # While we are in a split node, follow the splits
    while(!is.na(tr[index, "feat"])) {
      if(x[i, tr[index, "feat"]] <= tr[index, "split"]) {
        index <- tr[index, "index_left"]
      } else {
        index <- tr[index, "index_right"]
      }
    }
    
    y[i] <- tr[index, "mclass"]
  }
  
  return(y)
}

# tree.grow.bag
#
# @param x numeric dataframe without missing values containing the data to grow the tree on
# @param y vector with equal length as x containg 0, 1 values indicating the class of the rows in x
# @param nmin int minimum number of observations for a node to be allowed to split
# @param minleaf int minimum number of observations for a leaf node. Only splits that obey this value will be considered
# @param nfeat int number of features to be considered at each split
# @param m int number of trees to grow for the bag
# @return list of tree objects
#
# Grows m classification trees to classify the given data
# nmin, minleaf and nfeat can be used to restrict the growing process and prevent overfitting
tree.grow.bag <- function(x, y, nmin, minleaf, nfeat, m) {
  # Combine x and y in single dataframe, so they are linked when sampling
  x$class <- y
  r <- list()
  
  # Grow m trees on different bootstrap samples of the data
  for (i in 1:m) {
    bootstrap <- x[sample(nrow(x), replace=TRUE), ]
    r[[i]] <- tree.grow(bootstrap[, names(bootstrap) != "class"], bootstrap[, "class"], nmin, minleaf, nfeat)
  }
  
  r
}

# tree.classify.bag
#
# @param x numeric dataframe with similar features as the tree. These rows will be classified
# @param trs list of tree object from tree.grow.bag, used to classify the rows
# @return vector with the classification for each row of x
#
# Classifies new rows using a list of classification trees
tree.classify.bag <- function(x, trs) {
  # Predict class label with each tree in the forest
  # Results in a matrix where each row is the predictions for a single value of x, with columns representing trees from the forest
  preds <- sapply(trs, tree.classify, x=x)
  
  # For each value of x, get the majority vote of the trees
  apply(preds, 1, getmode)
}

#########################
# Auxilliary functions #
#########################

# impurity.gini
#
# @param x 0,1 vector
# @return num gini-index
#
# Calculates the impurity of a 0,1 vector by using the gini-index
impurity.gini <- function(x) {
  nodes.total <- length(x)
  nodes.0 <- length(which(x == 0))
  
  p.0 <- nodes.0 / nodes.total
  gini_index <- p.0 * (1 - p.0)
  
  gini_index
}

# split.best
#
# @param x numeric vector representing a single feature
# @param y 0,1 vector classifying the rows of x
# @param minleaf int minimum number of observations in either side of the split for it to be considered
# @return list containing the best split and its quality
#
# Calculate best split for a column x, class labels y and an optional minleaf constraint on the split
split.best <- function(x, y, minleaf=0) {
  # Splits are between two subsequent values
  x.sorted <- sort(unique(x))
  splits <- (x.sorted[1:length(x.sorted)-1]+x.sorted[2:length(x.sorted)])/2
  
  # Filter down to only acceptable splits, adhering to the minleaf constraint
  split.allowed <- function(split) {
    l <- y[x <= split]
    r <- y[x > split]
    
    length(l) >= minleaf && length(r) >= minleaf
  }
  splits <- Filter(split.allowed, splits)
  
  # Calculate impurity reduction of each split 
  x.impurity <- impurity.gini(y)
  split.quality <- function(split) {
    l <- y[x <= split]
    r <- y[x > split]
    
    l.impurity <- impurity.gini(l)
    r.impurity <- impurity.gini(r)
    
    x.impurity - (length(l)/length(x)*l.impurity) - (length(r)/length(x)*r.impurity)
  }
  splits.quality <- sapply(splits, split.quality)

  # Return split with maximum quality
  split.max = which.max(splits.quality)
  list(split=splits[split.max], quality=splits.quality[split.max])
}

# getmode
#
# @param x vector
# @return mode of x
#
# Utility to get the mode(most occuring element) from a vector, from https://www.tutorialspoint.com/r/r_mean_median_mode.htm
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

############
# Analysis #
############

n <- strtoi(readline(prompt="How many iterations of analysis would you like? This could take a couple minutes per iteration. n="))

if (n > 0) {
  # Train and test sets
  eclipse_train <- read.csv("eclipse-metrics-packages-2.0.csv", sep = ";")
  eclipse_test <- read.csv("eclipse-metrics-packages-3.0.csv", sep = ";")
  
  # Select columns, thus excluding AST features
  col_sub <- c("FOUT_avg", "FOUT_max", "FOUT_sum", "MLOC_avg", "MLOC_max", "MLOC_sum", "NBD_avg", "NBD_max", "NBD_sum", "PAR_avg", "PAR_max", "PAR_sum", "VG_avg", "VG_max", "VG_sum", "NOF_avg", "NOF_max", "NOF_sum", "NOM_avg", "NOM_max", "NOM_sum", "NSF_avg", "NSF_max", "NSF_sum", "NSM_avg", "NSM_max", "NSM_sum", "ACD_avg", "ACD_max", "ACD_sum", "NOI_avg", "NOI_max", "NOI_sum", "NOT_avg", "NOT_max", "NOT_sum", "TLOC_avg", "TLOC_max", "TLOC_sum", "NOCU", "pre")
  eclipse_train_sub <- eclipse_train[, col_sub]
  eclipse_train_class <- as.numeric(eclipse_train[, "post"] > 0)
  
  # stats
  #
  # @param r vector containing a classification of the eclipse_test set
  # @return num list of true positive count, true negative count, false positive count, false negative count, accuracy, precision and recall
  #
  # Compute accuracy, precision, recall
  stats <- function(r) {
    TN <- 0
    FP <- 0
    FN <- 0
    TP <- 0
    
    eclipse_test_class <- as.numeric(eclipse_test[, "post"] > 0)
    
    for (i in 1:length(r)) {
      if (eclipse_test_class[i] == 0) {
        if (r[i] == 0) {
          TN <- TN + 1
        } else {
          FP <- FP + 1
        }
      } else {
        if (r[i] == 0) {
          FN <- FN + 1
        } else {
          TP <- TP + 1
        }
      }
    }
    
    accuracy <- (TP +TN) / (TN + FN + FP + TP)
    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    
    return(list(TP=TP,TN=TN,FP=FP,FN=FN,accuracy=accuracy,precision=precision,recall=recall))
  }
  
  eclipse_stats <- list(n)
  eclipse_bagging_stats <- list(n)
  eclipse_forest_stats <- list(n)
  
  for (i in 1:n) {
    cat("Analysis iteration #", i, " out of ", n, "\n", sep="")
    
    # Single tree
    cat("Growing single tree...\n")
    eclipse_tree <- tree.grow(eclipse_train_sub, as.logical(eclipse_train_class), 15, 5, 41)
    cat("Classifying single tree...\n")
    eclipse_result <- tree.classify(eclipse_test, eclipse_tree)
    eclipse_stats[[i]] <- stats(eclipse_result)
    cat("Single tree finished!\n")
    
    # Bagging
    cat("Growing bagging...\n")
    eclipse_tree_bagging <- tree.grow.bag(eclipse_train_sub, as.logical(eclipse_train_class), 15, 5, 41, 100)
    cat("Classifying bagging...\n")
    eclipse_bagging_result <- tree.classify.bag(eclipse_test, eclipse_tree_bagging)
    eclipse_bagging_stats[[i]] <- stats(eclipse_bagging_result)
    cat("Bagging finished!\n")
    
    # Random forest
    cat("Growing random forest...\n")
    eclipse_tree_forest <- tree.grow.bag(eclipse_train_sub, as.logical(eclipse_train_class), 15, 5, 6, 100)
    cat("Classifying random forest...\n")
    eclipse_forest_result <- tree.classify.bag(eclipse_test, eclipse_tree_forest)
    eclipse_forest_stats[[i]] <- stats(eclipse_forest_result)
    cat("Random forest finished!\n")
  }
}
