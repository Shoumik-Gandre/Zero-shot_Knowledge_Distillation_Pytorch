python tabular.py `
    --num-samples 12000 `
    --batch-size 200 `
    --student-path './models/students/resmlp-resmlp-mnist.pt' `
    --teacher-path './models/teachers/tabular-resmlp.pt' `
    --synthetic-data-path './data/synthetic/tabular-resmlp/' `
    --real-data-path './data/real/' `
    --train-teacher `
    # --train-student `
    # --synthesize-data `
