# AI / ML project collection

## 1. Generative AI
### What is Generative AI?
It is used for generating some new contents or creative stories even images. That also makes it popular and unique in many areas.
![flow](https://www.google.com/url?sa=i&url=https%3A%2F%2Fprakat.com%2Fdata-analytics%2Fquick-guide-to-ai-gai-chatgpt-and-others%2F&psig=AOvVaw0ikGdpGDnCzowLlBq3uEx2&ust=1725606461811000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCNCu3I2fq4gDFQAAAAAdAAAAABAE)
![circle](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.linkedin.com%2Fpulse%2Fgenerative-ai-future-content-abhinay-panday-k9u6f&psig=AOvVaw0ikGdpGDnCzowLlBq3uEx2&ust=1725606461811000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCNCu3I2fq4gDFQAAAAAdAAAAABAT)
### How to achieve (algorithms)?
#### Task Specific Gen AI
1. Generative Adversarial Network(GAN)
一個(generator)負責生成資料，一個(discriminator)負責判別輸入進來的是真實資料還是假資料，如下圖
![GAN diagram](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/11/11/ML-6149-image025.jpg)
source pic: AWS-GAN
2. Diffusion model
Diffusion Models通過連續添加Gaussian noise來破壞訓練數據，然後學習透過反轉這個加noising的過程來恢復數據。訓練後，我們可以透過將隨機抽樣的noise通過學習得到的denoising過程來生成數據。
3. Variational Autoencoder(VAE)
透過encoder將輸入的圖片壓縮到很小，稱作Bottleneck，再透過decoder將圖片還原成原本大小，其學習目標是讓原始跟重建後的圖片一模一樣。
![VAE](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*PyUPcuHK9Nf_1X1f.png)

不過其實VAE學習的其實就是假設其為高斯分布將其學習去做平均值跟標準差，所以產生出來得圖片看起來尚可而已。事實上沒錯，其實VAE產生出來的圖片相較於GAN來說模糊很多，但是也因為VAE認真去學這個標準差和平均值，所以我們可以透過調控ϵ，生成特定的圖片。
![VAE生圖](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*VNhJVipZkRMVkY1r8LR9wg.png)

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

### Where can we use it?
1. RAG
2. Prompt engineering
3. Fine-tuning


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
