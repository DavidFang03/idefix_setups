def generate_particles(
    rmin,
    rmax,
    num_r,
    thetamin,
    thetamax,
    num_theta,
    betamin,
    betamax,
    sizemin,
    sizemax,
    num_k,
):

    particles = []
    n = 0

    for i in range(num_r):
        for j in range(num_theta):
            for k in range(num_k):
                # Prevent division by zero if a dimension has only 1 step
                r0 = rmin + (rmax - rmin) * i / (num_r - 1) if num_r > 1 else rmin
                theta0 = (
                    thetamin + (thetamax - thetamin) * j / (num_theta - 1)
                    if num_theta > 1
                    else thetamin
                )
                beta = (
                    betamin + (betamax - betamin) * k / (num_k - 1)
                    if num_k > 1
                    else betamin
                )
                size = (
                    sizemin + (sizemax - sizemin) * k / (num_k - 1)
                    if num_k > 1
                    else sizemin
                )

                # We also store the indices (i, j, k) because floating-point
                # numbers can have precision issues when filtering directly!
                particles.append(
                    {
                        "id": n,
                        "r_idx": i,
                        "theta_idx": j,
                        "k_idx": k,
                        "r": r0,
                        "theta": theta0,
                        "beta": beta,
                        "size": size,
                    }
                )
                n += 1

    return particles


# --- Test Setup Parameters ---
particles_list = generate_particles(
    rmin=1.0,
    rmax=5.0,
    num_r=10,
    thetamin=0.0,
    thetamax=360.0,
    num_theta=12,
    betamin=0.1,
    betamax=0.5,
    sizemin=2.0,
    sizemax=10.0,
    num_k=5,
)

print(f"Generated {len(particles_list)} particles successfully.\n")
