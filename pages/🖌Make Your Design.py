import streamlit as st
import torch
import torch.nn as nn
from rembg import remove

from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image
from IPython.display import Image as display_image

# Instance Normalization을 위한 mean과  std 계산

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# Adaptive Instance Normalization

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    # 평균(mean)과 표준편차(std)를 이용하여 정규화 수행
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    # 정규화 이후에 style feature의 statistics를 가지도록 설정
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# 인코더 정의

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(), # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(), # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(), # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(), # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(), # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(), # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(), # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(), # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(), # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU() # relu5-4
)

# 디코더 정의

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

decoder.eval()
vgg.eval()

vgg_path = 'C:/Users/최재원/Desktop/vgg_normalised.pth'
decoder_path = 'C:/Users/최재원/Desktop/decoder.pth'

decoder.load_state_dict(torch.load(decoder_path))
vgg.load_state_dict(torch.load(vgg_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)
decoder.to(device)

vgg = nn.Sequential(*list(vgg.children())[:31]) # ReLU4_1까지만 사용

# Adain Style Transfer Network

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4]) # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11]) # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18]) # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31]) # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image (중간 결과를 기록)
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image (최종 결과만 기록)
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    # 콘텐츠 손실(feature 값 자체가 유사해지도록)
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    # 스타일 손실(feature의 statistics가 유사해지도록)
    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        # 콘텐츠와 스타일 중 어떤 것에 더 많은 가중치를 둘지 설정 가능
        assert 0 <= alpha <= 1 # 0에 가까울수록 콘텐츠를 많이 살림
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t) # 결과 이미지
        g_t_feats = self.encode_with_intermediate(g_t)

        # 콘텐츠 손실과 스타일 손실을 줄이기 위해 두 개의 손실 값 반환
        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s

# Style Transfer 구현

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

# 이미지 전처리

def test_transform(size=512):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

content_tf = test_transform()
style_tf = test_transform()


#streamlit
st.set_page_config("TTok-Tak", "🎨")

st.title('TTok-Tak')
st.subheader('나만의 타투 스티커를 한번에')

main_img = Image.open(str('C:/Users/최재원/Desktop/main.png'))
st.image(main_img, width = 700)

st.sidebar.title('🎨 Main Menu')

# glide
multi_select = st.multiselect('타투 소재를 선택해 주세요',
                              ['강아지', '꽃', '별', '달', '해골', '고래', '사자', '뱀', '하트', '고양이'])
if multi_select == ['강아지']:
    prompt = "an tatto design of a dog"
    batch_size = 1
    guidance_scale = 3.0

elif multi_select == ['꽃']:
    prompt = "an tatto design of a flower"
    batch_size = 1
    guidance_scale = 3.0

elif multi_select == ['별']:
    prompt = "an tatto design of a star"
    batch_size = 1
    guidance_scale = 3.0

elif multi_select == ['달']:
    prompt = "an tatto design of a moon"
    batch_size = 1
    guidance_scale = 3.0

elif multi_select == ['해골']:
    prompt = "an tatto design of a skeleton"
    batch_size = 1
    guidance_scale = 3.0

if multi_select == ['고래']:
    prompt = "an tatto design of a whale"
    batch_size = 1
    guidance_scale = 3.0

elif multi_select == ['사자']:
    prompt = "an tatto design of a lion"
    batch_size = 1
    guidance_scale = 3.0

elif multi_select == ['뱀']:
    prompt = "an tatto design of a snake"
    batch_size = 1
    guidance_scale = 3.0

elif multi_select == ['하트']:
    prompt = "an tatto design of a heart"
    batch_size = 1
    guidance_scale = 3.0

elif multi_select == ['고양이']:
    prompt = "an tatto design of a cat"
    batch_size = 1
    guidance_scale = 3.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997

multi_selection = st.multiselect('원하는 스타일 세 가지 선택하세요',
                              ['펜드로잉', '고흐', '민화', '재패니즈', '팝아트', '피카소', '모네'])

if len(multi_selection) == 3:
    col1, col2, col3 = st.columns(3)
    with col1:
        save_opt = st.radio('사진을 저장하시겠습니까?', ('네', '아니오'))

        if st.button('도안 생성하기'):
            with col2:

                from PIL import Image
                from IPython.display import display
                import torch as th

                from glide_text2im.download import load_checkpoint
                from glide_text2im.model_creation import (
                    create_model_and_diffusion,
                    model_and_diffusion_defaults,
                    model_and_diffusion_defaults_upsampler
                )

                # This notebook supports both CPU and GPU.
                # On CPU, generating one sample may take on the order of 20 minutes.
                # On a GPU, it should be under a minute.

                has_cuda = th.cuda.is_available()
                device = th.device('cpu' if not has_cuda else 'cuda')

                # Create base model.
                options = model_and_diffusion_defaults()
                options['use_fp16'] = has_cuda
                options['timestep_respacing'] = '100'  # use 100 diffusion steps for fast sampling
                model, diffusion = create_model_and_diffusion(**options)
                model.eval()
                if has_cuda:
                    model.convert_to_fp16()
                model.to(device)
                model.load_state_dict(load_checkpoint('base', device))
                print('total base parameters', sum(x.numel() for x in model.parameters()))

                # Create upsampler model.
                options_up = model_and_diffusion_defaults_upsampler()
                options_up['use_fp16'] = has_cuda
                options_up['timestep_respacing'] = 'fast27'  # use 27 diffusion steps for very fast sampling
                model_up, diffusion_up = create_model_and_diffusion(**options_up)
                model_up.eval()
                if has_cuda:
                    model_up.convert_to_fp16()
                model_up.to(device)
                model_up.load_state_dict(load_checkpoint('upsample', device))
                # print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

                ##############################
                # Sample from the base model #
                ##############################

                # Create the text tokens to feed to the model.
                tokens = model.tokenizer.encode(prompt)
                tokens, mask = model.tokenizer.padded_tokens_and_mask(
                    tokens, options['text_ctx']
                )

                # Create the classifier-free guidance tokens (empty)
                full_batch_size = batch_size * 2
                uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
                    [], options['text_ctx']
                )

                # Pack the tokens together into model kwargs.
                model_kwargs = dict(
                    tokens=th.tensor(
                        [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
                    ),
                    mask=th.tensor(
                        [mask] * batch_size + [uncond_mask] * batch_size,
                        dtype=th.bool,
                        device=device,
                    ),
                )


                # Create a classifier-free guidance sampling function
                def model_fn(x_t, ts, **kwargs):
                    half = x_t[: len(x_t) // 2]
                    combined = th.cat([half, half], dim=0)
                    model_out = model(combined, ts, **kwargs)
                    eps, rest = model_out[:, :3], model_out[:, 3:]
                    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    eps = th.cat([half_eps, half_eps], dim=0)
                    return th.cat([eps, rest], dim=1)


                # Sample from the base model.
                model.del_cache()
                samples = diffusion.p_sample_loop(
                    model_fn,
                    (full_batch_size, 3, options["image_size"], options["image_size"]),
                    device=device,
                    clip_denoised=True,
                    progress=True,
                    model_kwargs=model_kwargs,
                    cond_fn=None,
                )[:batch_size]
                model.del_cache()

                ##############################
                # Upsample the 64x64 samples #
                ##############################

                tokens = model_up.tokenizer.encode(prompt)
                tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
                    tokens, options_up['text_ctx']
                )

                # Create the model conditioning dict.
                model_kwargs = dict(
                    # Low-res image to upsample.
                    low_res=((samples + 1) * 127.5).round() / 127.5 - 1,

                    # Text tokens
                    tokens=th.tensor(
                        [tokens] * batch_size, device=device
                    ),
                    mask=th.tensor(
                        [mask] * batch_size,
                        dtype=th.bool,
                        device=device,
                    ),
                )

                # Sample from the base model.
                model_up.del_cache()
                up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
                up_samples = diffusion_up.ddim_sample_loop(
                    model_up,
                    up_shape,
                    noise=th.randn(up_shape, device=device) * upsample_temp,
                    device=device,
                    clip_denoised=True,
                    progress=True,
                    model_kwargs=model_kwargs,
                    cond_fn=None,
                )[:batch_size]
                model_up.del_cache()

                # tensor to image
                from torchvision.transforms import ToPILImage
                import torch

                img_type = torch.squeeze(up_samples)
                tf_toPILImage = ToPILImage()
                img_PIL_from_Tensor = tf_toPILImage(img_type)
                # print(type(img_PIL_from_Tensor))

                st.image(img_PIL_from_Tensor)

                with col3:
                    for i in range(0, 4):
                        if i > 0:
                            style_path = 'C:/Users/최재원/Desktop/image/' + multi_selection[i - 1] + '.jpg'
                            content = content_tf(img=img_PIL_from_Tensor)
                            style = style_tf(Image.open(str(style_path)))

                            style = style.to(device).unsqueeze(0)
                            content = content.to(device).unsqueeze(0)
                            with torch.no_grad():
                                output = style_transfer(vgg, decoder, content, style, alpha=0.7)
                            output = output.cpu()

                            # 이미지 저장 후 업로드
                            save_image(output[0], 'C:/Users/최재원/Desktop/image/output.png')
                            img_PIL_from_Tensor = Image.open('C:/Users/최재원/Desktop/image/output.png').convert('RGB')

                            # 배경제거
                            # input = img_PIL_from_Tensor
                            # img_PIL_from_Tensor = remove(input)

                        # 인생네컷
                        if i == 0:
                            frame = Image.open(str('C:/Users/최재원/Desktop/frame1.png'))
                            frame = frame.resize((250, 1000))
                            file_img = img_PIL_from_Tensor
                            file_img = file_img.resize((170, 170))
                            img_PIL_from_Tensor = file_img

                        else:
                            img_PIL_from_Tensor = img_PIL_from_Tensor.resize((170, 170))
                        frame.paste(img_PIL_from_Tensor, (40, 57 + i * 190))

                    st.image(frame)

                    if save_opt == '네':
                        frame.save('C:/Users/최재원/Desktop/TTok-Tak/TTok-Tak.png')





