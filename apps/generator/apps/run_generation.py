"""SparkSubmit entrypoint for the generation job.

A stable file path for ``SparkSubmitOperator.application`` that delegates to the installed genpm
package. Works whether genpm is provided by the pinned wheel (prod) or a live mount / --py-files
(dev). All flags (``generate --conf-json ...``) are parsed by ``genpm.modelling.__main__.main``.
"""

from genpm.modelling.__main__ import main

if __name__ == "__main__":
    main()
