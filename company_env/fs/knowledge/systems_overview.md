# Company systems overview

- **Expense Portal** (`expense_portal`): web app for expense reports. Reimbursements are only paid for
  reports whose status is `submitted`. Sessions expire; sign in again if you are sent to the login page.
- **File share**: inbox/ holds employee-provided documents, knowledge/ holds policies and procedures,
  directory/ holds the employee directory, out/ is where operators write outputs.
Use `list_systems` to see exact URLs and `browser_login` to sign in. Credentials are handled by the platform;
never type or store passwords yourself.
