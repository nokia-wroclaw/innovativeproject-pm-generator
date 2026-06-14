"""SparkSubmit entrypoint for the preprocessing job.

A stable file path for ``SparkSubmitOperator.application`` that delegates to the installed genpm
package. Works whether genpm is provided by the pinned wheel (prod) or a live mount / --py-files
(dev). All flags (``--conf-json`` ...) are parsed by ``genpm.preprocessing.__main__.main``.
"""

from genpm.preprocessing.__main__ import main

if __name__ == "__main__":
    main()
