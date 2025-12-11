//! Bloom filter benchmarks
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ian_core::BloomFilter;

fn bench_bloom_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bloom Add");
    
    for size in [1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched(
                || BloomFilter::new(size, 0.01),
                |mut bf| {
                    bf.add(black_box(b"test_item"));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_bloom_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bloom Check");
    
    for size in [1000, 10000, 100000].iter() {
        let mut bf = BloomFilter::new(*size, 0.01);
        for i in 0..*size {
            bf.add(format!("item_{}", i).as_bytes());
        }
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &bf, |b, bf| {
            b.iter(|| {
                black_box(bf.maybe_contains(black_box(b"test_item")));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bloom_add, bench_bloom_check);
criterion_main!(benches);
