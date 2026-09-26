"""Post-production finish that takes the "digital / AI" edge off: one grade over every shot (real and generated),
gentle highlight roll-off, halation (red glow around highlights, like film), a soft vignette, and fine luma grain
that is identical across all footage so generated and real shots read as one camera."""


def look_filter(level):
    if level == "off":
        return ""
    if level == "film":
        grade = ("eq=contrast=1.06:saturation=0.86,"
                 "curves=all='0/0.035 0.25/0.235 0.75/0.765 1/0.965',"
                 "colorbalance=rs=-0.04:bs=0.05:rh=0.05:bh=-0.04,"
                 "gblur=sigma=0.35")
        halation, vignette, grain = 0.16, 0.55, 9
    else:  # subtle
        grade = ("eq=contrast=1.04:saturation=0.93,"
                 "curves=all='0/0.02 0.5/0.5 1/0.98',"
                 "colorbalance=rs=-0.02:bs=0.03:rh=0.03:bh=-0.02")
        halation, vignette, grain = 0.08, 0.4, 5
    return (f"{grade},split[lk_a][lk_b];"
            f"[lk_b]curves=all='0/0 0.72/0 1/1',gblur=sigma=14,colorchannelmixer=rr=1:gg=0.35:bb=0.2[lk_h];"
            f"[lk_a][lk_h]blend=all_mode=screen:all_opacity={halation},"
            f"vignette=angle={vignette},noise=c0s={grain}:c0f=t+u")
