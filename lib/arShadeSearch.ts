import { normalizeCatalogText } from '@/lib/catalogMatch';
import { isArPaintZone } from '@/store/makeupStore';
import type { Product, Shade } from '@/types';

export const AR_SHADE_SEARCH_LIMIT = 12;

export type ArShadeSearchHit = {
  product: Product;
  shade: Shade;
  score: number;
  arReady: boolean;
  blockReason: 'ok' | 'no-hex' | 'look-only' | 'no-zone';
};

export type ArShadeSearchResult =
  | { kind: 'idle' }
  | { kind: 'bad-intensity' }
  | { kind: 'none'; name: string; intensity?: number }
  | { kind: 'ok'; name: string; intensity?: number; hits: ArShadeSearchHit[] };

const HEX = /^#[0-9A-Fa-f]{6}$/;
const TRAILING_PCT = /^(.*?)\s+(-?\d{1,3})\s*%$/;

export function shadeHasArColor(shade: Shade): boolean {
  return HEX.test(shade.hex.trim());
}

export function arShadeBlockReason(product: Product, shade: Shade): ArShadeSearchHit['blockReason'] {
  if (!product.makeupZone) return 'no-zone';
  if (!isArPaintZone(product.makeupZone)) return 'look-only';
  if (!shadeHasArColor(shade)) return 'no-hex';
  return 'ok';
}

function haystacks(product: Product, shade: Shade): string[] {
  return [
    shade.name,
    product.brand,
    product.name,
    product.nameKo,
    product.searchKeyword,
    `${product.brand} ${product.name} ${shade.name}`,
    `${product.brand} ${product.nameKo} ${shade.name}`,
  ];
}

function scoreFields(query: string, product: Product, shade: Shade): number {
  const q = normalizeCatalogText(query);
  if (q.length < 1) return 0;
  let best = 0;
  for (const field of haystacks(product, shade)) {
    const n = normalizeCatalogText(field);
    if (!n) continue;
    if (n === q) best = Math.max(best, 1000);
    else if (n.startsWith(q)) best = Math.max(best, 800);
    else if (n.includes(q)) best = Math.max(best, 500);
  }
  return best;
}

function rankHits(products: Product[], query: string): ArShadeSearchHit[] {
  const scored: ArShadeSearchHit[] = [];
  for (const product of products) {
    for (const shade of product.shades) {
      const score = scoreFields(query, product, shade);
      if (score <= 0) continue;
      const blockReason = arShadeBlockReason(product, shade);
      scored.push({ product, shade, score, arReady: blockReason === 'ok', blockReason });
    }
  }
  scored.sort((a, b) => b.score - a.score || a.shade.name.localeCompare(b.shade.name, 'ko'));
  const seen = new Set<string>();
  const unique: ArShadeSearchHit[] = [];
  for (const hit of scored) {
    const key = `${hit.product.id}:${hit.shade.id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(hit);
    if (unique.length >= AR_SHADE_SEARCH_LIMIT) break;
  }
  return unique;
}

function mergeHits(primary: ArShadeSearchHit[], secondary: ArShadeSearchHit[]): ArShadeSearchHit[] {
  const seen = new Set(primary.map((hit) => `${hit.product.id}:${hit.shade.id}`));
  const extra = secondary.filter((hit) => {
    const key = `${hit.product.id}:${hit.shade.id}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return [...primary, ...extra].slice(0, AR_SHADE_SEARCH_LIMIT);
}

/** Catalog-only shade search. Never invents a hex. Intensity is 0–1 when a trailing percent is valid. */
export function searchArShades(raw: string, products: Product[]): ArShadeSearchResult {
  const text = raw.trim();
  if (!text) return { kind: 'idle' };

  const pct = text.match(TRAILING_PCT);
  if (pct) {
    const amount = Number(pct[2]);
    if (!Number.isFinite(amount) || amount < 0 || amount > 100) return { kind: 'bad-intensity' };
  }

  const fullHits = rankHits(products, text);
  if (!pct) {
    return fullHits.length ? { kind: 'ok', name: text, hits: fullHits } : { kind: 'none', name: text };
  }

  const name = pct[1].trim();
  const intensity = Number(pct[2]) / 100;
  if (!name) return { kind: 'none', name: text, intensity };

  const nameHits = rankHits(products, name);
  const hits = mergeHits(fullHits, nameHits);
  if (!hits.length) return { kind: 'none', name, intensity };
  return { kind: 'ok', name, intensity, hits };
}

