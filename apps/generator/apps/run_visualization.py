"""SparkSubmit entrypoint for the dataset visualization job.

A stable file path for ``SparkSubmitOperator.application`` that delegates to the installed genpm
package. Works whether genpm is provided by the pinned wheel (prod) or a live mount / --py-files
(dev). Subcommands (e.g. ``dataset``) and flags flow through argv to
``genpm.raw_vis.__main__.main``.
"""

from genpm.raw_vis.__main__ import main

if __name__ == "__main__":
    main()
