# Image Captioning
Image Captioning with [BLIP model](https://arxiv.org/abs/2201.12086) and deployed using [Gradio](https://github.com/gradio-app/gradio).

## Information

Ho Chi Minh City University of Science

Image and Video Processing Advanced - Assoc. Prof. Lý Quốc Ngọc

K31 - Master of Science - Group: CHOICES

No. | Student ID | Student Name
--- | :---: | ---
1 | 19127027 | Võ Hoàng Bảo Duy
2 | 19127094 | Phạm Ngọc Thiên Ân
3 | 19127292 | Nguyễn Thanh Tình

## How to run the deploy
1. Clone repository.
<pre>git clone https://github.com/ngthtinh99/ImageCaptioning.git</pre>
2. Install Python (Python 3.7 - 3.9 is required for supporting Pytorch).
3. Install necessary libraries.
<pre>pip install torch torchvision gradio timm fairscale transformers</pre>
4. Run the deploy, the first time downloading the model would take about 5 minutes, the next time would not need to reload.
<pre>python app.py</pre>
5. Browse the deploy on Localhost via the link http://localhost:7860, or the Public link generated in Command prompt.
6. Enjoy 🙂
