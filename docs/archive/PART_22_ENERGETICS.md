> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 22 — External energy ingestion

MolScope does not estimate potential, electrostatic, van der Waals, or
binding energies from coordinates alone. Instead it accepts a CSV energy table
with `trajectory_id`, `time_ps`, and one or more finite engine-produced energy
columns. Values are joined to frames by exact trajectory/time identity.

Missing energy rows are an error, never zero-filled or interpolated. Binding
free-energy and decomposition workflows remain separate methods that require
their own assumptions, input files, and validation.
