import gradio as gr
import os
import eval

dir_path = os.path.abspath('')
examples = [
        [os.path.join(dir_path, "dataset/images/00000675_(2).jpg"), os.path.join(dir_path,"dataset/sky/pexels-pixabay-2150.jpg")],
        [os.path.join(dir_path, "dataset/images/00000677.jpg"), os.path.join(dir_path,"dataset/sky/pexels-faik-akmd-1025469.jpg")],
        [os.path.join(dir_path, "dataset/images/00000792.jpg"), os.path.join(dir_path,"dataset/sky/pexels-felix-mittermeier-956981.jpg")],
        [os.path.join(dir_path, "dataset/images/00000833.jpg"), os.path.join(dir_path,"dataset/sky/pexels-shay-wood-574116.jpg")]
]


demo = gr.Interface(
    fn=eval.infernece,
    inputs=[
        gr.Image(type="numpy", sources='upload', label='original image'),
        gr.Image(type="numpy", sources='upload', label='sky image'),
        gr.CheckboxGroup(["Edge Detection", "HSV Thresholding"], label="Segmentation Methods", info="Choose one function"),\
    ],
    outputs=[
        gr.Image(type="numpy", label='predicted mask'),
        gr.Image(type="numpy", label='final image')
    ],
    title='Sky Replacement Demo',
    examples=examples,
)

if __name__ == '__main__':
    demo.launch(share=True)