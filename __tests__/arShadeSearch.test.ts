import { searchArShades, shadeHasArColor } from '@/lib/arShadeSearch';
import type { Product, Shade } from '@/types';

function shade(id: string, name: string, hex = '#C45A6A'): Shade {
  return { id, name, hex, measureMethod: 'estimated' };
}

function product(partial: Partial<Product> & Pick<Product, 'id' | 'name' | 'makeupZone' | 'shades'>): Product {
  return {
    brand: 'Brand',
    nameKo: partial.name,
    category: 'lip',
    texture: 'satin',
    priceKRW: 12000,
    volume: '1',
    description: '',
    keyIngredients: [],
    tags: [],
    imageUrl: '',
    searchKeyword: partial.name,
    ...partial,
  };
}

const catalog: Product[] = [
  product({
    id: 'uriage',
    brand: '유리아쥬',
    name: '스틱레브르 컬러드',
    makeupZone: 'lip',
    shades: [shade('eg70', '얼그레이 70%'), shade('rose', '로즈핑크')],
  }),
  product({
    id: 'romand',
    brand: '롬앤',
    name: '쥬시 래스팅 틴트',
    makeupZone: 'lip',
    shades: [shade('eg-rich', '35 얼그레이 리치'), shade('coral', '코랄 소르베')],
  }),
  product({
    id: 'blush',
    brand: '페리페라',
    name: '잉크 블러셔',
    makeupZone: 'blush',
    category: 'cheek',
    shades: [shade('coral-b', '코랄 치크', '#E08A7A')],
  }),
  product({
    id: 'mascara',
    brand: '클리오',
    name: '킬래쉬',
    makeupZone: 'mascara',
    category: 'eye',
    shades: [shade('black', '블랙', '#111111')],
  }),
  product({
    id: 'blank',
    brand: '테스트',
    name: '색 없는 립',
    makeupZone: 'lip',
    shades: [{ id: 'nohex', name: '빈색', hex: '', measureMethod: 'placeholder' }],
  }),
];

describe('searchArShades', () => {
  it('finds 얼그레이 in shade names without inventing a hex', () => {
    const result = searchArShades('얼그레이', catalog);
    expect(result.kind).toBe('ok');
    if (result.kind !== 'ok') return;
    expect(result.intensity).toBeUndefined();
    expect(result.hits.map((h) => h.shade.name)).toEqual(expect.arrayContaining(['얼그레이 70%', '35 얼그레이 리치']));
    expect(result.hits.every((h) => catalog.some((p) => p.shades.includes(h.shade)))).toBe(true);
  });

  it('keeps an exact option named 얼그레이 70% first, then splits intensity', () => {
    const result = searchArShades('얼그레이 70%', catalog);
    expect(result.kind).toBe('ok');
    if (result.kind !== 'ok') return;
    expect(result.intensity).toBeCloseTo(0.7);
    expect(result.hits[0]?.shade.name).toBe('얼그레이 70%');
    expect(result.hits[0]?.shade.hex).toBe('#C45A6A');
  });

  it('lists several 코랄 products at 50% coverage', () => {
    const result = searchArShades('코랄 50%', catalog);
    expect(result.kind).toBe('ok');
    if (result.kind !== 'ok') return;
    expect(result.intensity).toBeCloseTo(0.5);
    expect(result.hits.length).toBeGreaterThan(1);
    expect(result.hits.some((h) => h.product.makeupZone === 'blush')).toBe(true);
  });

  it('returns none for a missing shade', () => {
    expect(searchArShades('존재하지않는색 80%', catalog)).toEqual({
      kind: 'none',
      name: '존재하지않는색',
      intensity: 0.8,
    });
  });

  it('rejects intensity outside 0–100', () => {
    expect(searchArShades('얼그레이 150%', catalog).kind).toBe('bad-intensity');
    expect(searchArShades('얼그레이 -1%', catalog).kind).toBe('bad-intensity');
  });

  it('marks look-only and missing hex as not AR-ready', () => {
    const mascara = searchArShades('킬래쉬', catalog);
    const blank = searchArShades('빈색', catalog);
    expect(mascara.kind).toBe('ok');
    expect(blank.kind).toBe('ok');
    if (mascara.kind === 'ok') {
      expect(mascara.hits[0]?.arReady).toBe(false);
      expect(mascara.hits[0]?.blockReason).toBe('look-only');
    }
    if (blank.kind === 'ok') {
      expect(blank.hits[0]?.arReady).toBe(false);
      expect(blank.hits[0]?.blockReason).toBe('no-hex');
    }
    expect(shadeHasArColor({ id: 'x', name: 'x', hex: '#GG0000' })).toBe(false);
  });
});
