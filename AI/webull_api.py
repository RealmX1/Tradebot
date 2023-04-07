# not yet started; only copying related informaiton here for now 04/06

# I use the steps from Method 2 with the following CTraderBot class without issues.
from webull import webull, paper_webull
from _secrets import WEBULL_LOGIN_EMAIL, WEBULL_LOGIN_PWD, WEBULL_DEVICE_ID, \
                     WEBULL_TRADING_PIN, WEBULL_SECURITY_DID

class CTraderBot:
  def __init__(self, paper_trading: bool = False) -> None:
    self._webull = paper_webull() if (paper_trading) else webull()
    self._loggedin = False

  
  def login(self, use_workaround: bool = False) -> bool:
    wb = self._webull
    if (use_workaround):
      wb._set_did(WEBULL_SECURITY_DID)

    wb.login(username=WEBULL_LOGIN_EMAIL, password=WEBULL_LOGIN_PWD, device_name=WEBULL_DEVICE_ID)

    self._loggedin = wb.get_trade_token(WEBULL_TRADING_PIN)
    self._webull = wb

    return self._loggedin


def main():
  my_bot = CTraderBot(True)

  success = "Success!" if (my_bot.login()) else "Failed!"
  print(f"Logging into webull: {success}")


if __name__ == '__main__':
    main()
