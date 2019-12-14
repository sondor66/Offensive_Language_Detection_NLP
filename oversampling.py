def OverSampling(strat_train_set):
    df_majority = strat_train_set[strat_train_set['label'] == 0]
    df_minority = strat_train_set[strat_train_set['label'] == 1]
    major_count = len(df_majority)
    # oversample minority class
    df_minority_oversampled = resample(df_minority, 
                                 replace = True,              # sample with replacement
                                 n_samples = major_count,     # to match majority class 
                                 random_state = 42)    
     
         
    strat_train_set = pd.concat([df_majority, df_minority_oversampled])   # Combine majority class with oversampled minority class
    print("Train dataset calss distribution after Oversampling: \n", strat_train_set.label.value_counts())
  
    return strat_train_set
