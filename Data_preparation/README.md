# Why Data Preprocessing

千古名言:"Garbage in, Garbage out"，不論是在LLM的建模或是日後採用預訓練模型fine-tuning，都需要資料是高品質且收集企業內特定任務、場景的instruction tuning dataset是開始的第一步，資料的好壞決定日後訓練出的模型好壞。


# [Scaling Law](https://axk51013.medium.com/llm%E5%B0%88%E6%AC%84-%E8%BF%8E%E6%8E%A52024%E5%B9%B4-10%E5%80%8B%E5%BF%85%E9%A0%88%E8%A6%81%E6%90%9E%E6%87%82%E7%9A%84llm%E6%A6%82%E5%BF%B5-1-scaling-law-5f6a409d35c5)

模型性能可藉由參數量、Dataset大小、計算量預測

<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*qrM0R7T3BVkYSzTQY0udDw.png" alt="drawing" width="500"/>

## 但衍生出，兩個議題，

1. 收益: 固定compute resource，模型最好能表現到哪
2. 分配: 如何分配模型參數量跟資料集大小

## 他們基於三種方式來找到訓練LLM的Scaling Law：

1. 固定模型大小，變化訓練Data數量。
2. 固定計算量（浮點運算），變化模型大小。
3. 對所有實驗結果，直接擬合參數化 loss function。

## Chinchilla最大的貢獻更是在解決分配的問題，他們發現

1. 數據量（Tokens數）應該要約等於模型參數量的20倍
2. 並且數據量跟模型參數量要同比放大（Ex: 模型放大一倍，數據也要跟著增加一倍）

Scaling也是一個複雜的過程，期至少包含三個階段:

1. Cold start: 一開始模型小，資料量太少，呈現怎麼訓練都沒有幫助
2. Scaling: 正常的scaling週期
3. Plateau: 能力天花板，可能是dataset品質、架構設計天花板、本身任務有無法減少的錯誤

<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*iwLBNOIlfyxIki2kcc673Q.png" alt="drawing" width="500"/>

# [Data Quality](https://axk51013.medium.com/llm%E5%B0%88%E6%AC%84-%E8%BF%8E%E6%8E%A52024%E5%B9%B4-10%E5%80%8B%E5%BF%85%E9%A0%88%E8%A6%81%E6%90%9E%E6%87%82%E7%9A%84llm%E6%A6%82%E5%BF%B5-2-good-data-is-all-you-need-1e9e760c016a)

# [instruction tuning dataset](https://axk51013.medium.com/llm-10%E5%A4%A7%E8%A7%80%E5%BF%B5-3-%E5%BF%AB%E9%80%9F%E5%BB%BA%E9%80%A0%E8%87%AA%E5%B7%B1%E5%80%8Binstruction-tuning-dataset-ab391eba61e5)

Discuss: 如何finetune出一個公司內部解決特定任務的LLM?

1. Continual pretraining dataset: 讓LLM多讀相關領域的資料，通常會做到>10B以上
2. Instruction (fine)tuning dataset: input, otput包含指導LLM「當你面對特定用戶輸入時，你應該怎麼回應、怎麼解決」。

## 四種常見的收集instruction tuning data

1. Human annotate data（真人專家標注資料）
2. Human AI collaborate annotate data（真人與AI合作標注資料）
3. Distill from Large model (ex: GPT4)（用超強大模型生成）
4. Bootstrap from small or weak model (ex: llama2 7B)（用小模型一步一步生成）

