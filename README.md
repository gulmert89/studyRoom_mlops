# Study 1: Machine Learning Engineering for Production (MLOps) Specialization
## 1. Course One: Introduction to Machine Learning in Production
### 1.1 Week 1: Overview of the ML Lifecycle and Deployment
#### 1.1.1 Steps of an ML Project
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
#### 1.1.2 Key Challenges
* **Data drift (`X`)**
    * Distribution X changes, but mapping from X --> Y doesn't change.
    * E.g. new celebrity becomes much more well-known and starts to appear on voice search more.
* **Concept Drift (`X --> Y`)**
    * Definition of Y changes as X changes.
    * E.g. At Covid-19, a lot of online purchasing flagged as fraud (`false positive`) since people normally didn't shop online that much.
* Sometimes changes happen gradually as in language (new words or phrases get popular), and sometimes it happens suddenly as in online shopping during Covid-19.
* Software Engineering Issues
    * **Checklist of questions:**
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
#### 1.1.3 Deployment Patterns
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
        * if AI is not confident, send it to human. This is also useful for developing better models.
    5. Full Automation
        * You don't have to go to full automation all the time. You can stop before getting it if you like.
#### 1.1.4 Monitoring
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
#### 1.1.5 Pipeline Monitoring
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
#### 1.1.6 Quiz
* **Q:** You have built and deployed an anti-spam system that inputs an email and outputs either 0 or 1 based on whether the email is spam. Which of these will result in either concept drift or data drift?
* **A:** Spammers trying to change the wording used in emails to get around your spam filter.
#### 1.1.7 Week 1 References
* [Concept and Data Drift](https://towardsdatascience.com/machine-learning-in-production-why-you-should-care-about-data-and-concept-drift-d96d0bc907fb) (not read yet)
* [Monitoring ML Models](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/) (not read yet)
* [A Chat with Andrew on MLOps: From Model-centric to Data-centric](https://www.youtube.com/watch?v=06-AZXmwHjo) (not watched yet)
* [Konstantinos, Katsiapis, Karmarkar, A., Altay, A., Zaks, A., Polyzotis, N., … Li, Z. (2020). Towards ML Engineering: A brief history of TensorFlow Extended (TFX)](http://arxiv.org/abs/2010.02013) (not read yet)
* [Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2020). Challenges in deploying machine learning: A survey of case studies.](http://arxiv.org/abs/2011.09926) (not read yet)
* [Sculley, D., Holt, G., Golovin, D., Davydov, E., & Phillips, T. (n.d.). Hidden technical debt in machine learning systems. Retrieved April 28, 2021, from Nips.c](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) (not read yet)
### 1.2 Week 2: 
#### 1.2.1 Modeling Overview
* As hinted in the _side note_ in 1.1.1, it could be more useful to collect high quality data (`data-centric AI development`) instead of solely focusing on neural network architectures (`model-centric AI development`). Of course it's not about collecting more and more data either. It is about improving the data in the most efficient possible way.
#### 1.2.2 Key Challenges
* _"Because machine learning is such an empirical process, being able to go through the training/error analysis/model+hyperparameters+Data loop many times very quickly, is key to improving performance."_
* After doing enough iterations, one last thing to do could be performing a richer error analysis to audit performance.
* For many years, the engineering team focused on the performance on dev/test sets, which failed on business metrics & project goals. After doing fine on both training and dev/test sets, it's crucial to focus on these project goals as well.
#### 1.2.3 Why Low Average Error Isn't Good Enough
* Performance on disproportionately important examples!
    * User may forgive the slightly irrelevant results of the web search query `apple pie recipes`, but when the query is `reddit`, user has a crystal clear intent on this navigational query. Thus, if dev/test set performance is good in overall but the model messes up on this little group of navigational queries, it could be a serious problem. You could put more emphesis on this little but important group but it doesn't solve the entire problem.
* Performance on key slices of the dataset
    * Make sure not to discriminate by ethnicity, gender, location, language or other protected attributes. (Loan approval model)
    * Be careful to treat fairly all major user, retailer, and product categories. (Product recommendation model)
* Rare classes
    * Skewed data distribution: 99% negative data points, 1% positive data points. If you just `print("negative")`, you could be 99% correct, which could be the dumbest thing ever. You need to focus on, for example, rare diseases in a medical dataset. A rare case could be extremely important and fatal to be ignored, even though these cases don't hurt the averate test set performance.
* **You need to go beyond the test set and think like a product owner!**
#### 1.2.4 Establish a Baseline
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
#### 1.2.5 Tips for Getting Started
* Literature search to see what's possible (courses, blogs, open-source projects)
* Find open-source implmenetations if available.
* Don't obsess with the SoTA algorithms. A reasonable algo with good data will often outperform a great algo w/ no so good data.
* Just get started and iterate more!
* Should you take into account deployment constraints when picking a model?
    * Yes, if a baseline is already established and goal is to build & deploy.
    * No, if the purpose is to establish a baseline and determine what is possible and might be worth pursuing.
* Sanity-check for code & algorithm
    * Try to overfit a small training dataset before training on a large one.
#### 1.2.6 