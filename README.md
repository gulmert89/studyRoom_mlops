 # Study 1: `Machine Learning Engineering for Production (MLOps) Specialization`
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
### 1.2 Week 2: 
#### 1.2.1 
* ...loading...