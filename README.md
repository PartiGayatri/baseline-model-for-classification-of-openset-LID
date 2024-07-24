# baseline-model-for-classification-of-openset-LID

In this project, we have designed a baseline model for the classification of open-set language identification (LID). We leveraged 20 languages from the Voxlingua107 dataset, categorizing 10 languages as in-set and the remaining 10 as out-of-set.

**Methodology:**

Dataset and Embeddings:

We utilized the ECAPA-TDNN architecture from SpeechBrain, a pre-trained deep learning model, to generate audio embeddings from the speech samples. These embeddings represent the speech data in a high-dimensional space, capturing essential characteristics relevant to language identification.

**First Stage - One-Class SVM (OCSVM):**

A One-Class SVM is employed as the first stage. It is trained solely on speech data from known (in-set) languages. The OCSVM is expected to generate a boundary around the feature space of the in-set languages, ensuring that speech samples not belonging to the in-set languages are mapped outside the OCSVM boundary.

**Second Stage - Standard SVM Classifier:**

We take the same embeddings from in-set speech samples and pass them to a standard n-class SVM classifier. This SVM, trained on labelled speech data from the 10 in-set languages, classifies the samples into their respective languages based on the informative embeddings. It is further used to calculate language-specific thresholds for the classification of out-of-set LID.

**Evaluation:**

The effectiveness of this multi-stage approach will be evaluated using appropriate metrics like accuracy, precision, and recall. Our aim is to analyze the impact of the OCSVM stage in improving the open-set accuracy of the final language classification by the SVM.

By implementing this two-stage classification system, we seek to enhance the performance of open-set language identification and provide more accurate language-specific classification thresholds for out-of-set languages.
