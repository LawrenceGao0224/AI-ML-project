# AI / ML project collection

## 1. Generative AI
### What is Generative AI?
It is used for generating some new contents or creative stories even images. That also makes it popular and unique in many areas.
<img src="https://prakat.com/wp-content/uploads/2023/05/AI-chart.png" alt="drawing" width="500"/>
<img src="https://media.licdn.com/dms/image/v2/D4D12AQElpBj-AH3Fdw/article-inline_image-shrink_1000_1488/article-inline_image-shrink_1000_1488/0/1715826672831?e=1730937600&v=beta&t=sAyNkjE_MEE1v7i19v-9FktzPhNtWwKwNkAqaQONW5A" alt="circle" width="500"/>

### How to achieve (algorithms)?
#### Task Specific Gen AI
1. Generative Adversarial Network(GAN)
一個(generator)負責生成資料，一個(discriminator)負責判別輸入進來的是真實資料還是假資料，如下圖: (來源:AWS-GAN)
<img src="https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/11/11/ML-6149-image025.jpg" alt="GAN" width="500"/>

3. Diffusion model
Diffusion Models通過連續添加Gaussian noise來破壞訓練數據，然後學習透過反轉這個加noising的過程來恢復數據。訓練後，我們可以透過將隨機抽樣的noise通過學習得到的denoising過程來生成數據。
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png" alt="diffusion" width="500"/>

4. Variational Autoencoder(VAE)
透過encoder將輸入的圖片壓縮到很小，稱作Bottleneck，再透過decoder將圖片還原成原本大小，其學習目標是讓原始跟重建後的圖片一模一樣。
<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/0*PyUPcuHK9Nf_1X1f.png" alt="VAE" width="500"/>

不過其實VAE學習的其實就是假設其為高斯分布將其學習去做平均值跟標準差，所以產生出來得圖片看起來尚可而已。事實上沒錯，其實VAE產生出來的圖片相較於GAN來說模糊很多，但是也因為VAE認真去學這個標準差和平均值，所以我們可以透過調控ϵ，生成特定的圖片。
<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*VNhJVipZkRMVkY1r8LR9wg.png" alt="VAE生圖" width="500"/>

4. Flow Model

#### General GAI
1. The Generative Pre-Trained Transformer (GPT)
2. GPT-2
3. GPT-3
4. LLama from Meta
5. PaLM from Google
6. BLOOM
7. BERT from Google

### Generative AI Workflow
1. Data gathering
2. Data preprocessing and cleaning
3. Model Architecture selecting
4. Model Implementing
5. Model training
6. Evaluating and Optimizing
7. Fine-tuning and Iterate

### How can we achieve it? Tools: PyTorch, python ,Colab, Jupyter Notebook...
1. PyTorch and Tensorflow
2. Jupyter Notebook
3. HuggingFace
4. Pandas
5. OpenCV
6. Scikit-learn
7. AWS, Terraform, GitHub

### Most widely use Techniques?
1. RAG(Retrieval-Augmented Generation): 大型語言模型無法針對所需特定克制內容作回應，RAG可以有效地將所需資料成為輸入來源，當使用者針對該資料做出提問，LLM可以適當的回應其問題
如: PDFReader
![RAG](https://s4.tenten.co/learning/content/images/2024/06/rag-preprocessing-customized-mllms.png) 
3. Prompt engineering: 下簡單的指令讓LLM回答所需問題，簡單使用、符合成本效益，但是回答不一致、客製化有限、跟模型相依性大
如: AI girlfirend
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*Luk7ON3ZAv_ja9yO.png" alt="drawing" width="500"/>
4. Fine-tuning: 讓LLM學習新的或特定領域知識，就像更新手機APP一樣，更新內部的資料庫，可以良好的客製化、增加準確度，但成本較高、要有良好的資料集，且須要有技術能力
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*HJqbHWpar5tR9TB7.png" alt="drawing" width="500"/>

### [Use case](https://www.projectpro.io/article/artificial-intelligence-project-ideas/461)
1. Resume Parser AI Project
2. 2. Fake News Detector Project in AI
3. Translator App
4. Instagram Spam Detection
5. Object Detection System 
6. Animal Species Prediction
7. Pneumonia Detection with Python
8. Teachable Machine
9. Autocorrect Tool
10. Fake Product Review Identification
### Intermidiate Level project
1. Price Comparison Application
2. Ethnicity Detection Model
3. Traffic Prediction
4. Age Detection Model
5. Image to Pencil Sketch App
6. Hand Gesture Recognition Model 
7. Text Generation Model
8. Colour Detection
9. Sign Language Recognition App with Python
10. Detecting Violence in Videos
### Top 10 AI Frameworks and Libraries
| Category | Frameworks/Libraries |
| :---: | :---: |
| Traditional Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Deep Learning | TensorFlow, PyTorch |
| Computer Vision | OpenCV |
| Natural Language Processing | Hugging Face |
| Large Language Models | OpenAI, LangChain, LlamaIndex |

## 2. Fine-tune and model evaluation: how to fine tune and evalute
## 3. data preprocessing 
one-hot encoding, label data normalize

## 4. feature engineering
## 5. BQML(GCP big query ML)
## 6. Computer vision
## 7. GPUs
## 8. Speech recongnition
## 9. Search recommendation
## 10. Advertising
## 11. GBDT
