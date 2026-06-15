"""SparkSubmit entrypoint for the data similarity job.

A stable file path for ``SparkSubmitOperator.application`` that delegates to the installed genpm
package. Works whether genpm is provided by the pinned wheel (prod) or a live mount / --py-files
(dev). All flags are parsed by ``genpm.data_similarity.__main__.main``.
"""

from genpm.data_similarity.__main__ import main

if __name__ == "__main__":
    main()
