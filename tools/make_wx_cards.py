#!/usr/bin/env python3
"""生成微信分享用的方形缩略图（docs/wx-*.jpg）。

微信不读 Open Graph：链接卡片的缩略图来自页面 body 顶部的 <img>，
且要求 ≥300x300、JPG、HTTPS 可直接访问。现有 og-*.png 是 1200x630 宽图，
被微信裁成方形后文字全丢，所以另出一套方图。

用法：python3 tools/make_wx_cards.py
依赖：playwright（复用页面内嵌的 Space Grotesk / Space Mono 字体）
"""
import re, os, subprocess, sys
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, 'docs')
TMP = '/tmp/wxcard.html'
SIZE = 500          # CSS 像素，DSF=2 出 1000x1000

CARDS = [
    ('index', '01', '灵巧手',   '入门全攻略', '选型 · 仿真 · 数据闭环 · 供应链'),
    ('body',  '02', '人形本体', '全景分析',   '关节 · BOM · 中美格局 · 宇树深拆'),
    ('brain', '03', '具身大脑', '全景分析',   'VLA · 世界模型 · 数据金字塔'),
]

# 复用页面里内嵌的字体，保证与正文同一套排版
src = open(os.path.join(DOCS, 'index.html'), encoding='utf-8').read()
faces = re.findall(r'@font-face\s*\{[^}]*\}', src, re.S)
if not faces:
    sys.exit('未能从 index.html 提取 @font-face')
FONTS = '\n'.join(faces)

TPL = """<!doctype html><meta charset="utf-8"><style>
%(fonts)s
:root{--paper:#fff;--ink:#121212;--muted:#6f6f6f;--faint:#a1a1a1;--line:#dedede;--red:#e00016;
 --sans:"Space Grotesk","PingFang SC","Noto Sans SC",sans-serif;--mono:"Space Mono","PingFang SC",monospace;}
*{margin:0;padding:0;box-sizing:border-box}
body{width:%(size)spx;height:%(size)spx;background:var(--paper);font-family:var(--sans);
 color:var(--ink);position:relative;overflow:hidden;
 background-image:radial-gradient(var(--line) 1px,transparent 1px);background-size:19px 19px;}
.pad{position:absolute;inset:0;padding:46px 44px;display:flex;flex-direction:column;}
.rule{width:52px;height:5px;background:var(--ink);}
.kick{display:flex;align-items:center;gap:9px;margin-top:20px;
 font-family:var(--mono);font-size:14px;letter-spacing:0.14em;color:var(--muted);}
.kick::before{content:"";width:11px;height:11px;border-radius:50%%;background:var(--red);flex:none;}
.mid{margin-top:auto;margin-bottom:auto;}
.t1{font-size:76px;font-weight:700;letter-spacing:-0.02em;line-height:1.05;}
.t2{font-size:27px;font-weight:700;color:var(--red);margin-top:10px;letter-spacing:0.02em;}
.sub{margin-top:22px;font-size:16px;color:var(--muted);line-height:1.6;}
.foot{display:flex;align-items:center;gap:12px;font-family:var(--mono);
 font-size:14px;color:var(--faint);letter-spacing:0.06em;}
.foot b{color:var(--ink);font-size:17px;}
.foot .ln{flex:1;height:1px;background:var(--line);}
</style>
<div class="pad">
  <div class="rule"></div>
  <div class="kick">具身智能三部曲</div>
  <div class="mid">
    <div class="t1">%(t1)s</div>
    <div class="t2">%(t2)s</div>
    <div class="sub">%(sub)s</div>
  </div>
  <div class="foot"><b>%(num)s</b><span class="ln"></span><span>robot.jasonlin.tech</span></div>
</div>"""

with sync_playwright() as p:
    b = p.chromium.launch()
    for slug, num, t1, t2, sub in CARDS:
        html = TPL % dict(fonts=FONTS, size=SIZE, num=num, t1=t1, t2=t2, sub=sub)
        open(TMP, 'w', encoding='utf-8').write(html)
        pg = b.new_page(viewport={'width': SIZE, 'height': SIZE}, device_scale_factor=2)
        pg.goto('file://' + TMP)
        pg.wait_for_timeout(600)
        png = f'/tmp/wx-{slug}.png'
        pg.screenshot(path=png)
        pg.close()
        out = os.path.join(DOCS, f'wx-{slug}.jpg')
        subprocess.run(['magick', png, '-quality', '88', out], check=True)
        kb = os.path.getsize(out) // 1024
        dim = subprocess.run(['identify', '-format', '%wx%h', out],
                             capture_output=True, text=True).stdout
        print(f'wx-{slug}.jpg  {dim}  {kb}KB')
    b.close()
