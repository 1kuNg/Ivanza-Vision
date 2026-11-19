"""Imagens sintéticas e reproduzíveis para ensinar classificação Few-Shot / SKU.

Não há resultados fingidos: as imagens apenas alimentam as APIs normais
/api/sku/register e /api/sku/predict usando checkpoints reais do DINO.
"""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from math import cos, sin, pi
from random import Random

from PIL import Image, ImageDraw, ImageFilter

SKU_DEMOS = {
    'embalagens': {
        'title': 'Embalagens: frasco x caixa',
        'description': 'Duas embalagens de formatos diferentes e uma lata desconhecida.',
        'classes': ('Frasco azul', 'Caixa laranja'),
        'unknown': 'Lata verde',
        'explanation': 'Reconheça o mesmo tipo de embalagem em poses diferentes; teste a rejeição da lata.',
    },
    'fixadores': {
        'title': 'Peças: porca x arruela',
        'description': 'Dois tipos de fixadores e um parafuso que não foi cadastrado.',
        'classes': ('Porca sextavada', 'Arruela'),
        'unknown': 'Parafuso',
        'explanation': 'Compare dois formatos metálicos e teste um fixador que o sistema nunca viu.',
    },
}


def _polygon(cx, cy, radius, n, rotation=0):
    return [(int(cx+radius*cos(rotation+2*pi*i/n)),
             int(cy+radius*sin(rotation+2*pi*i/n))) for i in range(n)]


def _draw_bottle(d, x, y, scale, bg):
    w, h = 132*scale, 228*scale
    def box(x0,y0,x1,y1):
        return [round(x+x0*scale),round(y+y0*scale),round(x+x1*scale),round(y+y1*scale)]
    d.ellipse(box(2,211,132,236),fill='#aab4bd')
    d.rounded_rectangle(box(43,-6,89,19),radius=6*scale,fill='#183856',outline='#12293d',width=max(1,round(3*scale)))
    d.polygon([(round(x+28*scale),round(y+38*scale)),(round(x+43*scale),round(y+15*scale)),(round(x+89*scale),round(y+15*scale)),(round(x+105*scale),round(y+38*scale))],fill='#25679a')
    d.rounded_rectangle(box(22,35,110,218),radius=15*scale,fill='#237ab6',outline='#15547f',width=max(1,round(3*scale)))
    d.rounded_rectangle(box(32,107,101,170),radius=8*scale,fill='#e6eef3')
    d.text((round(x+46*scale),round(y+124*scale)),'A-01',fill='#225170')
    d.rounded_rectangle(box(31,47,43,199),radius=5*scale,fill='#7fc6e0')


def _draw_box(d,x,y,scale):
    def box(x0,y0,x1,y1):
        return [round(x+x0*scale),round(y+y0*scale),round(x+x1*scale),round(y+y1*scale)]
    d.ellipse(box(-7,214,154,242),fill='#aeb6c1')
    d.polygon([(round(x),round(y+18*scale)),(round(x+117*scale),round(y+18*scale)),(round(x+142*scale),round(y+0*scale)),(round(x+22*scale),round(y))],fill='#f6b767')
    d.rectangle(box(0,18,116,224),fill='#e58c2f',outline='#9b5317',width=max(1,round(3*scale)))
    d.polygon([(round(x+116*scale),round(y+18*scale)),(round(x+142*scale),round(y)),(round(x+142*scale),round(y+203*scale)),(round(x+116*scale),round(y+224*scale))],fill='#b66822')
    d.rectangle(box(20,82,99,152),fill='#fff7e8',outline='#d4a66f',width=max(1,round(2*scale)))
    d.text((round(x+31*scale),round(y+110*scale)),'B-02',fill='#79441b')
    d.rectangle(box(55,18,66,78),fill='#f7bc7a')


def _draw_can(d,x,y,scale):
    def box(x0,y0,x1,y1):
        return [round(x+x0*scale),round(y+y0*scale),round(x+x1*scale),round(y+y1*scale)]
    d.ellipse(box(-7,211,140,240),fill='#aeb6c1')
    d.rectangle(box(4,25,128,219),fill='#36a16d')
    d.ellipse(box(4,7,128,43),fill='#a8b8bc',outline='#667d82',width=3)
    d.ellipse(box(5,192,127,232),fill='#277a54',outline='#225c43',width=3)
    d.rectangle(box(5,51,127,178),fill='#3ab87b')
    d.rounded_rectangle(box(17,93,114,157),radius=6*scale,fill='#d4f3dc')
    d.text((round(x+31*scale),round(y+120*scale)),'C-03',fill='#165c39')
    d.rectangle(box(22,51,33,177),fill='#79caa3')


def _draw_nut(d,cx,cy,scale,rotation,bg):
    r=108*scale
    points=_polygon(cx,cy,r,6,rotation)
    shadow=_polygon(cx+13*scale,cy+15*scale,r,6,rotation)
    d.polygon(shadow,fill='#b5c0c7')
    d.polygon(points,fill='#a1adb8',outline='#525f6e',width=max(2,round(6*scale)))
    d.polygon(_polygon(cx,cy,r*.76,6,rotation),fill='#ced5dd',outline='#718390',width=max(2,round(4*scale)))
    r2=41*scale
    d.ellipse((cx-r2,cy-r2,cx+r2,cy+r2),fill=bg,outline='#526170',width=max(2,round(5*scale)))


def _draw_washer(d,cx,cy,scale,bg):
    r=112*scale
    d.ellipse((cx-r+13,cy-r+13,cx+r+13,cy+r+13),fill='#b5c0c7')
    d.ellipse((cx-r,cy-r,cx+r,cy+r),fill='#aab8c4',outline='#546475',width=max(2,round(6*scale)))
    d.ellipse((cx-r+16,cy-r+16,cx+r-16,cy+r-16),outline='#e2e8ef',width=max(2,round(9*scale)))
    hole=65*scale
    d.ellipse((cx-hole,cy-hole,cx+hole,cy+hole),fill=bg,outline='#5d6d7a',width=max(2,round(5*scale)))


def _draw_bolt(d,cx,cy,scale):
    w=51*scale
    d.rounded_rectangle((cx-w,cy-15*scale,cx+w,cy+142*scale),
                        radius=9*scale,fill='#8595a2',outline='#53636f',width=4)
    for i in range(8):
        yy=cy+27*scale+i*14*scale
        d.line((cx-w,yy,cx+w,yy),fill='#dae1e7',width=max(1,round(3*scale)))
    d.polygon(_polygon(cx,cy-47*scale,86*scale,6,pi/6),fill='#aab7c4',outline='#596976',width=5)
    d.polygon(_polygon(cx,cy-47*scale,64*scale,6,pi/6),outline='#e2e7ee',width=4)


@lru_cache(maxsize=40)
def render_demo_image(demo: str, item: str, variant: int) -> bytes:
    """Produz PNG estável. Os exemplos são ilustrações, não fotos industriais reais."""
    if demo not in SKU_DEMOS:
        raise KeyError(demo)
    allowed = ('frasco','caixa','lata') if demo == 'embalagens' else ('porca','arruela','parafuso')
    if item not in allowed or not 0 <= variant <= 10:
        raise ValueError('Item de demonstração inválido.')

    width,height=640,420
    rng=Random(sum(ord(ch) for ch in f'{demo}:{item}')+variant*31)
    img=Image.new('RGB',(width,height))
    d=ImageDraw.Draw(img)
    for y in range(height):
        grey=int(226+y*.025+rng.randrange(-2,3))
        d.line((0,y,width,y),fill=(grey,min(grey+5,255),min(grey+10,255)))
    # Marcas discretas ajudam a perceber pequenas variações de enquadramento.
    d.line((26,370,614,370),fill='#d7e0e6',width=2)
    scale=1+(.025*(variant%3-1))
    dx=(variant%3-1)*12
    dy=(variant%2)*5
    if demo=='embalagens':
        left=260+dx
        top=77+dy
        if item=='frasco':
            _draw_bottle(d,left,top,scale,(230,235,240))
        elif item=='caixa':
            _draw_box(d,left-10,top,scale)
        else:
            _draw_can(d,left,top,scale)
    else:
        x,y=320+dx,202+dy
        bg=(232,237,242)
        if item=='porca':
            _draw_nut(d,x,y,scale,variant*.07,bg)
        elif item=='arruela':
            _draw_washer(d,x,y,scale,bg)
        else:
            _draw_bolt(d,x,y,scale)
    # A fonte discreta é legenda da própria simulação; não entra na classificação.
    d.text((21,18),'IVANZA VISION LAB  |  AMOSTRA SINTETICA',fill='#536576')
    d.text((21,390),f'{demo.upper()}  /  AMOSTRA {variant+1}',fill='#6a7a88')
    output=BytesIO()
    img.save(output,format='PNG',optimize=True)
    return output.getvalue()


def demo_payload(name: str, upload):
    """Registra imagens no Store atual para o frontend usar o caminho real de inferência."""
    config=SKU_DEMOS[name]
    keys=('frasco','caixa','lata') if name=='embalagens' else ('porca','arruela','parafuso')
    classes=[]
    for label,key in zip(config['classes'],keys[:2]):
        classes.append(dict(name=label,photos=[
            upload(render_demo_image(name,key,variant)) for variant in (0,1,2)
        ]))
    tests=[
        dict(name=config['classes'][0],expected=config['classes'][0],
             photo=upload(render_demo_image(name,keys[0],4))),
        dict(name=config['classes'][1],expected=config['classes'][1],
             photo=upload(render_demo_image(name,keys[1],5))),
        dict(name=config['unknown'],expected='Desconhecido',
             photo=upload(render_demo_image(name,keys[2],6))),
    ]
    return dict(id=name,title=config['title'],description=config['description'],
                explanation=config['explanation'],classes=classes,tests=tests)
