# Study 1: Machine Learning Engineering for Production (MLOps) Specialization
## 1. Course One: Introduction to Machine Learning in Production
### 1.1. Week 1: Overview of the ML Lifecycle and Deployment
#### 1.1.1. Steps of an ML Project
* Two crucial points: Data drift & concept drift
    * E.g. children start to use Siri more, which is trained mostly on adults.
    * The factory gets darker but most of the training photos of products were under well-lit conditions.
* Steps:
    * **Scoping**
        * Define project
            * Decide what to work on. E.g. speech recognition for voice search.
            * Define key metrics (accuracy, latency, throughput, privacy etc.)
            * Define resources & timeline
    * **Data**
        * Define data and establish baseline.
        * Label and organize data.
            * Is the data labeled consistently?
    * **Modeling**
        * Select & train model
            * Code (algorithm/model)
            * Hyperparameters
            * Data
        * Perform error analysis
            * Pinpoint the parts to be improved.
        * Side note: Research/Academia mostly focuses on `code` & `hyperparameters` whereas product teams focus on `hyperparameters` and `data` and take action accordingly.
    * **Deployment**
        * Deploy in production
            * Edge/Mobile device or PC browser? 
        * Monitor & maintain the system
            * Concept/Data drift!
#### 1.1.2. Key Challenges
* **Data drift (`X`)**
    * Distribution X changes, but mapping from X --> Y doesn't change.
    * E.g. new celebrity becomes much more well-known and starts to appear on voice search more.
* **Concept Drift (`X --> Y`)**
    * Definition of Y changes as X changes.
    * E.g. At Covid-19, a lot of online purchasing flagged as fraud (`false positive`) since people normally didn't shop online that much.
* Sometimes changes happen gradually as in language (new words or phrases get popular), and sometimes it happens suddenly as in online shopping during Covid-19.
* **Software Engineering Issues**
    * Checklist of questions:
        * Realtime or batch?
            * For example, some of my projects work on ***batches*** since the jobs scheduled to start on specific hours to work on the most current data saved to the database.
        * Cloud vs. Edge/Browser
            * Factories mostly work on edge devices due to possible internet shortages.
        * Compute resources (CPU/GPU/Memory)
            * Sometimes you train your model on powerful GPUs but will it be used on such devices as well? Maybe you will deploy it on a small edge device. ***Is it as powerful as the GPUs in your cloud/local device?***
        * Latency, throughput (QPS)
            * Maybe your limits are `500ms` or `1000 queries per second`?
        * Logging
        * Security & Privacy
#### 1.1.3. Deployment Patterns
* Common cases:
    * New product/feature
    * Automate/assist with manual task
    * Replace previous ML system
* Key ideas:
    * Gradual ramp up with monitoring (Start with small traffic).
    * Rollback if not working.
* First Stage (`Shadow Mode`)
    * ML system shadows the human & runs in parallel. Human ***observes*** it firstly.
    * The system output shouldn't be used for any decisions during this phase.
* Second Stage (`Canary Deployment`)
    * *"Canary in a coal mine"*
    * Roll out for small fraction (say `5%`) of traffic initially.
    * Monitor the system and ramp up the traffic gradually.
* Third Stage (`Blue Green Deployment`)
    * First,
        * [`Phone Images`] ---> [`Router`] ---> Old/Blue Version
    * Later, **switch** to `Green` version.
        * [`Phone Images`] ---> [`Router`] ~~---> Old/Blue Version~~ ---> New/Green Version
    * Easy way to enable rollback
* Degrees of Automation
    1. Human Only
    2. Shadow Mode
    3. AI Assistance [`Human in the loop`]
        * e.g. only provides `bounding box` to human
    4. Partial Automation [`Human in the loop`]
        * if AI is not confident, sends it to human. This is also useful for developing better models.
    5. Full Automation
        * You don't have to go to full automation all the time. You can stop before getting it if you like.
#### 1.1.4. Monitoring
* You can monitor anything you want: Server load, fraction of non-null values, fraction of missing input values etc.
* Brainstorm the things that could go wrong.
* Brainstorm a few statistics/metrics that will detect the problem.
* It is ok to use many metrics initially and gradually remove the ones you find not useful.
* Example metrics to track (e.g. for a speech recognition system)
    * Software Metrics: Memory, compute, latency, throughput, server load
    * **Input Metrics (X)**: Average input length, average input volume, # of missing values, average image brightness or size or shape
    * **Output Metrics (Y)**: # of times `" "` (null), # of times user redoes the search, # of times user switches to typing, CTR (click-through rate)
* Just as ML modeling is iterative, so is deployment!
    * Deploy/Monitor ---> Traffic ---> Perform Analysis ---> Deploy/Monitor ---> ...
    * Choosing a right metric is an iterative process as well. Sometimes you discover a new problem after a few weeks and start to track it on a new metric, or sometimes a metric doesn't change at all and gives no insight/information and turns out to be useless.
* Common Practices
    * Set threshold for alarms
        * E.g. server load gets `> 90%`
        * Missing values get below *some point*.
* You will adapt metrics and thresholds over time.
* Then you will perform ***manual training*** (more common) or ***automatic training*** (used more in consumer software internet) if necessary.
#### 1.1.5. Pipeline Monitoring
* A sample "speech recognition" pipeline:
    1. [`Audio`] --->
    2. [`Voice Activity Detection (VAD)`] (A model that detects if user is speaking to the assistant or not.) --->
    3. [`Speech Recognition`] (A model that transcripts the audio it receives from `VAD`.) --->
    4. [`Transcript`]
* Here, `VAD` (which is on the device) gets the speech and clips it to reduce bandwidth sent to cloud and passes it to the `speech recognition` (which is on the cloud). However, if the performance of `VAD` decreases, it affects the performance of `Speech Recognition` system as well!
* Another "user profile" pipeline:
    1. [`User Data`] (e.g. clickstream) --->
    2. [`User Profiler Model`] (e.g. own a car?) --->
    3. [`Recommender System`] --->
    4. [`Product Recommendations`]
* Here, let's say for the question "User owns a car?", `user profiler model` provides `Y`, `N` or `UNKNOWN` to the `Recommender System`. If the `UNKNOWN` labels (which you monitor) start to increase, recommender system's performance gets worse over time, so do the recommended products'.
* How quickly do the data change?
    * User data generally has slower drift.
        * Exceptions: Covid-19, a holiday destination becomes popular all of a sudden etc.
    * Enterprise data (B2B applications) can shift fast.
        * A company start to use another raw material for their end product.
#### 1.1.6. Quiz
* **Q:** You have built and deployed an anti-spam system that inputs an email and outputs either 0 or 1 based on whether the email is spam. Which of these will result in either concept drift or data drift?
* **A:** Spammers trying to change the wording used in emails to get around your spam filter.
#### 1.1.7. Week 1 References
* [Concept and Data Drift](https://towardsdatascience.com/machine-learning-in-production-why-you-should-care-about-data-and-concept-drift-d96d0bc907fb) (not read yet)
* [Monitoring ML Models](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/) (not read yet)
* [A Chat with Andrew on MLOps: From Model-centric to Data-centric](https://www.youtube.com/watch?v=06-AZXmwHjo) (not watched yet)
* [Konstantinos, Katsiapis, Karmarkar, A., Altay, A., Zaks, A., Polyzotis, N., … Li, Z. (2020). Towards ML Engineering: A brief history of TensorFlow Extended (TFX)](http://arxiv.org/abs/2010.02013) (not read yet)
* [Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2020). Challenges in deploying machine learning: A survey of case studies.](http://arxiv.org/abs/2011.09926) (not read yet)
* [Sculley, D., Holt, G., Golovin, D., Davydov, E., & Phillips, T. (n.d.). Hidden technical debt in machine learning systems. Retrieved April 28, 2021, from Nips.c](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) (not read yet)
### 1.2. Week 2: Select and Train a Model
#### 1.2.1. Modeling Overview
* As hinted in the _side note_ in 1.1.1, it could be more useful to collect high quality data (`data-centric AI development`) instead of solely focusing on neural network architectures (`model-centric AI development`). Of course it's not about collecting more and more data either. It is about improving the data in the most efficient possible way.
#### 1.2.2. Key Challenges
* _"Because machine learning is such an empirical process, being able to go through the training/error analysis/model+hyperparameters+Data loop many times very quickly, is key to improving performance."_
* After doing enough iterations, one last thing to do could be performing a richer error analysis to audit performance.
* For many years, the engineering team focused on the performance on dev/test sets, which failed on business metrics & project goals. After doing fine on both training and dev/test sets, it's crucial to focus on these project goals as well.
#### 1.2.3. Why Low Average Error Isn't Good Enough
* Performance on disproportionately important examples!
    * User may forgive the slightly irrelevant results of the web search query `apple pie recipes`, but when the query is `reddit`, user has a crystal clear intent on this navigational query. Thus, if dev/test set performance is good in overall but the model messes up on this little group of navigational queries, it could be a serious problem. You could put more emphesis on this little but important group but it doesn't solve the entire problem.
* Performance on key slices of the dataset
    * Make sure not to discriminate by ethnicity, gender, location, language or other protected attributes. (Loan approval model)
    * Be careful to treat fairly all major user, retailer, and product categories. (Product recommendation model)
* Rare classes
    * Skewed data distribution: 99% negative data points, 1% positive data points. If you just `print("negative")`, you could be 99% correct, which could be the dumbest thing ever. You need to focus on, for example, rare diseases in a medical dataset. A rare case could be extremely important and fatal to be ignored, even though these cases don't hurt the averate test set performance.
* **You need to go beyond the test set and think like a product owner!**
#### 1.2.4. Establish a Baseline
* When you see the accuracies below, you could think that "We need to work on _low bandwidth_ accuracy!". However, if it is practically on HLP (Human Level Performance) level, you should think about improving _car noise_ speeches instead.

|Type|Accuracy|HLP|Improvement|
|:---|:---:|:---:|:---:|
|Clear Speech|94%|95%|1%|
|Car Noise|89%|93%|4%|
|People Noise|87%|89%|2%|
|Low Bandwidth|70%|70%|~0%|

* HLP is a less useful baseline for a structured data (such as spreadsheet/tabular data) than images, audios and text (user comments etc.).
* Ways to establish a baseline:
    * HLP
    * Literature search for SoTA/open source materials
    * Quick-and-dirty implementation
    * Performance of older system
* Baseline helps to indicate what might be possible. In some cases (such as HLP), it also gives a sense of what is irreducible error/Bayes error (an upper limit to your system's potential). So, we can be much more efficient in terms of prioritizing what to work on.
* Make this establishment in the beginning to save time!
#### 1.2.5. Tips for Getting Started
* Literature search to see what's possible (courses, blogs, open-source projects)
* Find open-source implmenetations if available.
* Don't obsess with the SoTA algorithms. A reasonable algo with good data will often outperform a great algo w/ no so good data.
* Just get started and iterate more!
* Should you take into account deployment constraints when picking a model?
    * Yes, if a baseline is already established and goal is to build & deploy.
    * No, if the purpose is to establish a baseline and determine what is possible and might be worth pursuing.
* Sanity-check for code & algorithm
    * Try to overfit a small training dataset before training on a large one.
#### 1.2.6. Error Analysis Example
* Collaborative tagging, training and deploying platform: [landing.ai (formerly LandingLens)](https://landing.ai/platform/)
* You can come up with common problems (in a speech recognition model) like "car noise", "people noise" etc. and listen random audio clips that were predicted wrong, and then thick under these problems in a regular spreadsheet. Think about your banner project, you can detect problematic products by creating such a spreadsheet and thick the possible problems.
* Iterative process: Examine tag examples <------> Propose tags
* Useful metric questions for each tag (error class):
    * "What fraction of errors has that tag?" e.g. 12% in all the classes
    * "Of all data with that tag, what fraction is misclasiffied?" e.g. all the data with "car noise", 18% is misclassified. It tells you how hard the examples with car noise are.
    * "What fraction of all the data has that tag?"
    * "How much room for improvement there is on data with that that?" e.g. measuring HLP.
 #### 1.2.7. Prioritizing What to Work on
 |Type|Accuracy|HLP|Gap to HLP|% of data|Raise in Avg. Acc.|
|:---|:---:|:---:|:---:|:---:|:---:|
|Clear Speech|94%|95%|1%|60%|0.60%|
|Car Noise|89%|93%|4%|4%|0.16%|
|People Noise|87%|89%|2%|30%|0.60%|
|Low Bandwidth|70%|70%|~0%|6%|~0%|
* Previously, we said that _"we should think about improving _car noise_ speeches instead"_ considering the gap to the HLP, but when we make a deeper analysis, we see that working on _people noise_ and _clear speech_ might increase the average accuracy by 0.6% individually. Thus, considering the **amount of data** is another crucial factor here!
* To prioritizing stuff, ask yourself:
    * How much room for improvement there is.
    * How frequently that category appears.
    * How easy to improve accuracy in that category is.
    * How important it is to improve that category.
* Thus, you don't have to collect more data for _not-that-vital_ categories and save time!
#### 1.2.8. Skewed Datasets
* When you have very skewed data, raw accuracy score is not much useful (remember `print("negative")` from the ***Section 1.2.3***).
* **Confusion Matrix** (with _recall_, _precision_ & _F1-score_) is much more useful!
    * **Precision:** [focuses on `FP`] A _not defected_ smartphone goes to human inspection since the model predicts it as _defected_. Not that crucial for the company.
    * **Recall:** [focuses on `FN`] A _defected_ smartphone goes into the market and gets sold as defected! It is ***crucial*** for the company!
#### 1.2.9. Performance Auditing
* Brainstorm the ways the system might go wrong (_speech recognition_ examples below):
    * Performance (e.g. accuracy) on subsets of data
        * Accuracy on ethnicity, gender
        * Accuracy on different devices
        * Prevalence of rude mis-transcriptions (e.g. GAN --> gun, gang)
    * How common are certain errors (e.g. FP, FN)
    * Performance on rare classes
* Establish metrics to assess performance against these issues on appropriate slices of data.
    * Mean accuracy for different genders & major accents.
    * Mean accuracy for different devices.
    * Check for prevalence of offensive words in the output.
* Get business/product ownder buy-in.
#### 1.2.10. Data-centric AI development
* The quality of the data is paramount. Use tools to improve data quality; this will allow multiple models to do well.
* Hold the code fixed & iteratively improve the data.
* In academy, they hold the data fixed and work on the models because a fixed dataset lets you perform benchmark tests on different models.
#### 1.2.11. A useful picture of data augmentation
![A useful picture of data augmentation](https://raw.githubusercontent.com/gulmert89/studyRoom_mlops/main/mlops_specialization/c2w2-ungraded-lab/images/a-useful-picture-of-data-augmentation.png)
* When we push the performance of a point up, the nearby points get affected more than the farther points.
#### 1.2.12. Data Augmentation
* What type of noise to be used? How loud shoud it be?
* Goal is to **create realistic examples** that:
    * the algorithm does poorly on,
    * but humans (or other baseline) do well on
* Checklist:
    * Does it sound realistic?
    * Is the X ---> Y mapping clear? (e.g. can humans recognize speech?)
    * Is the algorithm currently doing poorly on it?
#### 1.2.13. Can adding data hurt?
* For ***unstructured*** data problems, if:
    * the model is large enough (low bias),
        * e.g. increase café data from 20% to 50%. Yes, it changes the data distribuion $P(x)$ but if the model is large enough, it increases the performance unless the model is small. Then it hurts the non-café data performance.
    * the mapping X ---> Y is clear (e.g. given only the input X, humans can make accurate predictions.),
        * e.g. in Google street view images, there are hard cases on house numbers. It is quite hard to guess between the number $1$ and the letter $I$ in the images. Thus, creating synthetic images with $I$ could hurt the performance in this case since it is a corner case where a lot of house numbers don't contain letters. In this case, it is safer to guess $Is$ as $1$.
        * Adding a lot of new $Is$ may skew the dataset. 
    * then, **adding data rarely hurts accuracy**.
#### 1.2.14. Adding Features
* Restaurant recommendation example (**Structured** data problems):
* Our system recommends meat options to a vegetarian! We want to change that. Imagine the model takes `person` and `restaurant` examples and `recommends` a new restaurant to user. Also, you know that you can't changed the structured data here much, which are `person` and `restaurant`. They are kinda fix and data synthesising is not possible (are you going to create a fake restaurant ffs?). So, one thing to do that you can add a new feature to these data.
* What possible features to add?
    * Is person vegetarian (based on past orders)?
        * Doesn't need to be a discrete data. It could be the percentage of fruits ordered.
    * Does restaurant have vegetarian options (base on their menu)?
        * To detect it, you can use another model that reads the menu. Or some basic algorithm to detect it.
* There is a shift on filtering in recommendation systems from `Collaborative filtering` to `Content based filtering`.
    * **Collaborative filtering**: Recommends items which other profiles similar to you consume.
        * This has `cold-start` problem. How do you recommend a brand-new restaurant which doesn't have any stars, reviews or comments? If no one interacted with this item, it won't be recommended to anybody.
    * **Content based filtering**: Model analyses you and the item, then matches it according to similar interests.
        * It solves the `cold-start` problem.
        * but it requires a successful feature extraction from that brand-new item.
* ***Adding features*** is also an iterative process. The features are data as well! So, you add your features, train the model, do error analysis and observe the performance of the features you added, then iterate.
    * Error analysis can be harder if there is no good baseline (such as HLP) to compare to.
    * Error analysis, user feedback & benchmarking to competitors can all provide inspiration for features to add.
* In pre-deep learning era, the feature extraction and creation was more important. Now, the DL algorithms discover the features in images, sounds and texts pretty good. However, in ***structured*** data, it is still a thing.
#### 1.2.15. Experiment Tracking
* What to track:
    * Algorithm/code versioning
    * Dataset used
    * Hyperparameters
    * Results
* Tracking tools:
    * Text files (This doesn't scale well. C'mon!)
    * (Shared) Spreadsheet
    * Experiment tracking system
        * ``WandB``
        * ``Comet MLflow``
        * ``SageMaker Studio``
        * ``Landing.AI`` (on computer vision & manufacture systems)
* Desirable features:
    * Information needed to replicate results.
    * Experiment results, ideally with summary metrics/analysis
    * Perhaps also: Resource monitoring, visualization, model error analysis
#### 1.2.16. From Big Big Data to Good Data
* Try to ensure consistently high-quality data in all phases of the ML project lifecycle.
* Good data:
    * Covers important cases (good coverage of inputs X),
    * Is defined consistently (definition of labels U is unambiguous),
    * Has timely feedback from production data (distribution covers data drift & concept drift),
    * Is sized appropriately.
#### 1.2.17. Week 2 Optional References
* [Establishing a baseline](https://blog.ml.cmu.edu/2020/08/31/3-baselines/)
* [Error analysis](https://techcommunity.microsoft.com/t5/azure-ai/responsible-machine-learning-with-error-analysis/ba-p/2141774)
* [Experiment tracking](https://neptune.ai/blog/ml-experiment-tracking)
* [Brundage, M., Avin, S., Wang, J., Belfield, H., Krueger, G., Hadfield, G., … Anderljung, M. (n.d.). Toward trustworthy AI development: Mechanisms for supporting verifiable claims.](http://arxiv.org/abs/2004.07213v2)
* [Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2019). Deep double descent: Where bigger models and more data hurt](http://arxiv.org/abs/1912.02292)
### 1.3. Week 3: Data Definition and Baseline
#### 1.3.1. Why is data definition hard?