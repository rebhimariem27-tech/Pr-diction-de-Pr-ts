/*
  Run this script while connected as a database administrator or another
  principal allowed to grant permissions.

  Replace [FINPREDICT_APP_USER] with the database user mapped to SQL_APP_USERNAME
  in the backend .env file. This is a database principal, not the API key.
*/

USE [FinPredictTrackingDB];
GO

GRANT SELECT ON OBJECT::dbo.model_versions TO [FINPREDICT_APP_USER];
GRANT SELECT, INSERT ON OBJECT::dbo.loan_applications TO [FINPREDICT_APP_USER];
GRANT UPDATE (actual_decision, decision_at_utc, updated_at_utc)
    ON OBJECT::dbo.loan_applications TO [FINPREDICT_APP_USER];
GRANT SELECT, INSERT ON OBJECT::dbo.prediction_logs TO [FINPREDICT_APP_USER];
GRANT SELECT ON OBJECT::dbo.loan_outcome_events TO [FINPREDICT_APP_USER];
GRANT SELECT ON OBJECT::dbo.outcome_labels TO [FINPREDICT_APP_USER];
GRANT SELECT ON OBJECT::dbo.retraining_runs TO [FINPREDICT_APP_USER];
GO

/*
  Optional hardening after validating the application:

  DENY DELETE ON OBJECT::dbo.loan_applications TO [FINPREDICT_APP_USER];
  DENY DELETE ON OBJECT::dbo.prediction_logs TO [FINPREDICT_APP_USER];
*/
