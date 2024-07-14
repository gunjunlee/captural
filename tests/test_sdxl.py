import pytest

from captural import Captural

import torch
from diffusers import DiffusionPipeline

def test_tmp():

    x = torch.randn((4, 4)).cuda()
    y = torch.randn((4, 4)).cuda()

    def foo(x, y):
        f = poo(x, y)
        g = poo(x, poo(x,
                       y))
        return f + g

    def poo(x, y):
        result = (x
                  +
                  y) \
                  * 2
        return result

    def too(x, y):
        z = x
        x = y
        y = z
        return z

    def woo(x, y):
        z, w  = f, g = poo(x, y), poo(x, poo(x,
                       y))
        return f + g + z

    def zoo(x, y):
        params = {id(k): v for k, v in zip(x, y)}
        return params


    import dis
    print(dis.dis(foo))
    print()
    print(dis.dis(poo))
    print()
    print(dis.dis(too))
    print()
    print(dis.dis(woo))
    print()
    print(dis.dis(zoo))
    with Captural() as cap:
        zoo(x, y)

    cap.print_stats()


def test_stable_diffusion_xl_base_1_0_fp16():

    dtype = torch.float16
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    pipe.to(device)

    latent = torch.randn(2, 4, 128, 128, device=device, dtype=dtype)
    t = torch.tensor(10., device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(2, 77, 2048, device=device, dtype=dtype)
    added_cond_kwargs = {
        "text_embeds": torch.randn(2, 1280, device=device, dtype=dtype),
        "time_ids": torch.tensor([[1024., 1024.,    0.,    0., 1024., 1024.], [1024., 1024.,    0.,    0., 1024., 1024.]], device=device, dtype=dtype),
    }

    with Captural([
        ".*diffusers/models/resnet.*",
        ".*diffusers/models/transformers.*",
        ".*diffusers/models/unets.*",
        ".*diffusers/models/downsampling.*",
        ".*diffusers/models/upsampling.*",
        ".*diffusers/models/embeddings.*",
    ]) as cap:
    # with Captural(["."]) as cap:
        pipe.unet(
            latent, t, encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs
        )

    cap.print_stats()


def main():
    # import dis
    # print(dis.dis(test_tmp))
    # test_tmp()
    test_stable_diffusion_xl_base_1_0_fp16()


if __name__ == "__main__":
    main()