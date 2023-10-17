import pandas as pd
import os, pathlib
test_root = pathlib.Path('/home/jthomas/pycharm/JTtools/test_data/')

# for saving figures or wookbooks to look at
out_dir = pathlib.Path('/mnt/m/tmp')

def orthogonal_regression():

    from jttools.statistics import (
        OrthogonalRegression
    )
    fn = test_root/'xy_df.csv'
    df = pd.read_csv(fn)
    OR = OrthogonalRegression()
    orth = OR.tls_simple(df.x, df.y)
    print(orth)

    orthbs = OR.tls_bootstrap(df.x, df.y, nboot=100)
    print(orthbs)

    from jttools.plotting import plot_qq
    import matplotlib.pyplot as plt
    plot_qq(orth.residuals)
    plt.savefig(out_dir/'OR_qq.png', bbox_inches='tight')
    plt.close()

    orthbs.plot(bs_ci = (95, 80, 50))
    plt.legend()
    plt.savefig(out_dir / 'OR_ci_bs.png', bbox_inches='tight')
    plt.close()

    orth.plot()
    plt.savefig(out_dir/'OR_ci_normal.png', bbox_inches='tight')

orthogonal_regression()