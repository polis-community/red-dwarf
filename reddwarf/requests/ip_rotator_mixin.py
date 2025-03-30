import logging
from reddwarf.requests.adapters import CloudflareBypassHTTPAdapter
from requests import Session
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from typing import List, Optional
from requests_ip_rotator import ApiGateway, ip_rotator

# Set up logger
logger = logging.getLogger(__name__)

class IPRotatorMixin(Session):
    """
    A mixin for requests.Session that adds IP rotation capabilities using requests-ip-rotator.

    This mixin makes it easy to add IP rotation to any requests Session by automatically
    managing ApiGateway instances and cleanup. If AWS credentials aren't configured,
    it will fall back to regular request behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gateways = {}  # Store gateway objects by domain
        self._rotation_enabled = True  # Global flag to track if rotation is enabled

    def _check_aws_credentials(self, access_key_id: Optional[str] = None,
                              access_key_secret: Optional[str] = None) -> tuple[bool, str]:
        """
        Check if AWS credentials are available either from parameters or environment.

        Args:
            access_key_id: AWS access key ID
            access_key_secret: AWS access key secret

        Returns:
            Tuple[bool, str]: (True if credentials are available, reason if not)
        """
        try:
            # If credentials were provided directly, check those
            if access_key_id and access_key_secret:
                # Test creating a simple client with the provided credentials
                session = boto3.Session(
                    aws_access_key_id=access_key_id,
                    aws_secret_access_key=access_key_secret
                )
                # Just try to get the available regions to verify credentials
                session.get_available_regions('apigateway')
                return True, ""

            # Otherwise check if credentials are available from environment or config
            session = boto3.Session()
            # Check if credentials are available in the session
            credentials = session.get_credentials()
            if credentials is None:
                return False, "No AWS credentials found in environment or config files"

            # Verify the credentials work by testing a simple operation
            session.get_available_regions('apigateway')
            return True, ""

        except (NoCredentialsError, PartialCredentialsError) as e:
            return False, f"AWS credentials error: {str(e)}"
        except Exception as e:
            return False, f"Error checking AWS credentials: {str(e)}"

    def use_ip_rotation(
        self,
        site: str,
        regions: Optional[List[str]] = ip_rotator.DEFAULT_REGIONS,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        force: bool = False,
        require_manual_deletion: bool = False,
        verbose: bool = True,
        fallback_on_error: bool = True
    ) -> bool:
        """
        Enable IP rotation for the specified site.

        Args:
            site: The base URL of the site (e.g., "https://site.com")
            regions: AWS regions to use (defaults to ApiGateway's DEFAULT_REGIONS)
            access_key_id: AWS access key ID (defaults to environment variables)
            access_key_secret: AWS access key secret (defaults to environment variables)
            force: Force creation of new gateways even if they already exist
            require_manual_deletion: Whether gateways should persist after shutdown()
            verbose: Whether to print status messages
            fallback_on_error: If True, will fall back to regular requests on AWS errors

        Returns:
            bool: True if IP rotation was successfully enabled, False otherwise
        """
        # First check if AWS credentials are available
        credentials_available, reason = self._check_aws_credentials(access_key_id, access_key_secret)

        if not credentials_available:
            if fallback_on_error:
                logger.warning(f"Cannot enable IP rotation: {reason}")
                logger.warning("Falling back to regular requests without IP rotation")
                return False
            else:
                raise ValueError(f"AWS credentials not available: {reason}")

        try:
            # Create the gateway
            gateway = ApiGateway(
                site=site,
                regions=regions,
                access_key_id=access_key_id,
                access_key_secret=access_key_secret,
                verbose=verbose
            )

            # Start the gateway
            gateway.start(force=force, require_manual_deletion=require_manual_deletion)

            # Mount it to this session
            self.mount(site, gateway)

            # Store gateway for cleanup
            self._gateways[site] = gateway

            logger.info(f"IP rotation enabled for {site}")
            return True

        except Exception as e:
            if not fallback_on_error:
                # Re-raise the exception if fallback is disabled
                raise

            logger.warning(f"Failed to enable IP rotation for {site}: {str(e)}")
            logger.warning("Falling back to regular requests without IP rotation")
            return False

    def disable_ip_rotation(self, site: Optional[str] = None) -> None:
        """
        Disable IP rotation for a site or all sites if none specified.

        Args:
            site: The site to disable IP rotation for, or None for all sites
        """
        if site is not None:
            if site in self._gateways:
                try:
                    self._gateways[site].shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down gateway for {site}: {str(e)}")
                finally:
                    del self._gateways[site]
                    logger.info(f"IP rotation disabled for {site}")
        else:
            # Disable for all sites
            for site, gateway in list(self._gateways.items()):
                try:
                    gateway.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down gateway for {site}: {str(e)}")
                finally:
                    del self._gateways[site]

            logger.info("IP rotation disabled for all sites")

    def set_rotation_enabled(self, enabled: bool) -> None:
        """
        Globally enable or disable IP rotation.

        Args:
            enabled: True to enable rotation, False to disable
        """
        self._rotation_enabled = enabled
        logger.info(f"IP rotation globally {'enabled' if enabled else 'disabled'}")

        if not enabled:
            # Temporarily unmount all gateways but don't shut them down
            for site in self._gateways:
                self.mount(site, CloudflareBypassHTTPAdapter())
        else:
            # Re-mount all gateways
            for site, gateway in self._gateways.items():
                self.mount(site, gateway)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disable_ip_rotation()