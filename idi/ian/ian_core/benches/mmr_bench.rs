//! MMR benchmarks
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ian_core::MerkleMountainRange;

fn bench_mmr_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("MMR Append");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched(
                || {
                    let mut mmr = MerkleMountainRange::with_capacity(size);
                    for i in 0..size-1 {
                        mmr.append(format!("leaf_{}", i).as_bytes());
                    }
                    mmr
                },
                |mut mmr| {
                    mmr.append(black_box(b"new_leaf"));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_mmr_root(c: &mut Criterion) {
    let mut group = c.benchmark_group("MMR Root");
    
    for size in [100, 1000, 10000].iter() {
        let mut mmr = MerkleMountainRange::new();
        for i in 0..*size {
            mmr.append(format!("leaf_{}", i).as_bytes());
        }
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &mmr, |b, mmr| {
            b.iter(|| {
                black_box(mmr.get_root());
            });
        });
    }
    group.finish();
}

fn bench_mmr_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("MMR Proof");
    
    for size in [100, 1000, 10000].iter() {
        let mut mmr = MerkleMountainRange::new();
        for i in 0..*size {
            mmr.append(format!("leaf_{}", i).as_bytes());
        }
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &mmr, |b, mmr| {
            b.iter(|| {
                let idx = black_box(size / 2);
                black_box(mmr.get_proof(idx).unwrap());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mmr_append, bench_mmr_root, bench_mmr_proof);
criterion_main!(benches);
