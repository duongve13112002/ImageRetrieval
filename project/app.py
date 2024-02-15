import os
import numpy as np
from PIL import Image
import cv2
import glob
import gradio as gr
from source.main import ImageSearch
import gc
import torch
def inference(model,dataset,similarity_vectors,number_image,image_in):
    IS = ImageSearch(dataset, model,similarity_vectors)
    IS.indexing()
    res, _ = IS.retrieve_image(image_in, number_image)
    paths = list(np.array(res)[:, 0])
    paths = [str(p) for p in paths]
    torch.cuda.empty_cache()
    gc.collect()
    return gr.Gallery.update(value=paths)

def flip_image(x,angle):
    return gr.Image.update(value=x.rotate(angle))

models = ['VisionTransformer','BEiT','MobileViTV2','Bit','EfficientFormer','MobileNetV2','ResNet','EfficientNet']
datasets = ['All Database','oxbuild','paris']
similarity_function = ['cosine','TS_SS','euclidean']
current_model = models[1]
current_dataset = datasets[0]
def get_model_list():
    return models
def get_dataset_list():
    return datasets


css = """
.finetuned-diffusion-div div{
    display:inline-flex;
    align-items:center;
    gap:.8rem;
    font-size:1.75rem;
    padding-top:2rem;
}
.finetuned-diffusion-div div h1{
    font-weight:900;
    margin-bottom:7px
}
.finetuned-diffusion-div p{
    margin-bottom:10px;
    font-size:94%
}
.box {
  float: left;
  height: 20px;
  width: 20px;
  margin-bottom: 15px;
  border: 1px solid black;
  clear: both;
}
a{
    text-decoration:underline
}
.tabs{
    margin-top:0;
    margin-bottom:0
}
#gallery{
    min-height:20rem
}
.no-border {
    border: none !important;
}
 """

with gr.Blocks(css = css) as demo:
    #gr.Markdown("Image retrieval")
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Demo for retrieval project</h1>
              </div>
              <p>Image retrieval</p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column():
            model = gr.Dropdown(
                            choices=[k for k in get_model_list()],
                            label="Model",
                            value=current_model,
                        )
            with gr.Row():
                dataset = gr.Dropdown(
                            choices=[k for k in get_dataset_list()],
                            label="Dataset",
                            value=current_dataset,
                        )
                similarity_vectors = gr.Dropdown(
                            choices=[k for k in similarity_function],
                            label="Similarity Function",
                            value=similarity_function[0],
                        )
            
            image_in = gr.Image(source="upload",type='pil')
            with gr.Row():
                number_angle = gr.Slider(
                                            label="Angle", value=45, minimum=-180, maximum=180, step=0.1
                                        )
                number_image = gr.Slider(
                                        label="Number Images Retrival", value=8, minimum=1, maximum=64, step=1
                                    )
                
            with gr.Row():
                image_button = gr.Button("Rotate")
                find = gr.Button(value="Find")

            image_button.click(flip_image, inputs=[image_in,number_angle], outputs=image_in)
            
            
        gallery = gr.Gallery(label="Retrieved images", show_label=True, elem_id="gallery")
        inputs=[model,
            dataset,
            similarity_vectors,
            number_image,
            image_in,
        ]
        find.click(inference, inputs=inputs, outputs=gallery)

demo.queue().launch(share=True,debug=True)
demo.launch(enable_queue=True, server_name="0.0.0.0", server_port=7860)