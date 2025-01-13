#pragma once

#include <iostream>
#include <vector>
#include "simulation.h"

std::vector<PokerCard> royalFlush    = {PokerCard::Th, PokerCard::Jh, PokerCard::Qh, PokerCard::Kh, PokerCard::Ah};
std::vector<PokerCard> straightFlush = {PokerCard::_9h, PokerCard::Th, PokerCard::Jh, PokerCard::Qh, PokerCard::Kh};
std::vector<PokerCard> fourOfAKind   = {PokerCard::_9h, PokerCard::_9d, PokerCard::_9c, PokerCard::_9s, PokerCard::Kh};
std::vector<PokerCard> fullHouse     = {PokerCard::_9h, PokerCard::_9d, PokerCard::_9c, PokerCard::Kh, PokerCard::Ks};
std::vector<PokerCard> flush         = {PokerCard::_2c, PokerCard::_4c, PokerCard::_6c, PokerCard::_8c, PokerCard::Tc};
std::vector<PokerCard> straight      = {PokerCard::_9h, PokerCard::Td, PokerCard::Jh, PokerCard::Qs, PokerCard::Kh};
std::vector<PokerCard> wheel         = {PokerCard::Ah, PokerCard::_2c, PokerCard::_3s, PokerCard::_4d, PokerCard::_5c};
std::vector<PokerCard> broadway      = {PokerCard::Tc, PokerCard::Jh, PokerCard::Qs, PokerCard::Kh, PokerCard::Ah};
std::vector<PokerCard> threeOfAKind  = {PokerCard::_9h, PokerCard::_9d, PokerCard::_9c, PokerCard::Qs, PokerCard::Kh};
std::vector<PokerCard> twoPair       = {PokerCard::_9h, PokerCard::_9d, PokerCard::Kh, PokerCard::Ks, PokerCard::Ah};
std::vector<PokerCard> onePair       = {PokerCard::_9h, PokerCard::_9d, PokerCard::Qs, PokerCard::Kh, PokerCard::Ah};
std::vector<PokerCard> highCard      = {PokerCard::_9h, PokerCard::Jh, PokerCard::Qs, PokerCard::Kh, PokerCard::Ah};
std::vector<PokerCard> twoAces       = {PokerCard::Ac, PokerCard::Ad, PokerCard::_7s, PokerCard::_5h, PokerCard::_2h};
std::vector<PokerCard> threeAces     = {PokerCard::Ac, PokerCard::Ad, PokerCard::Ah, PokerCard::_7s, PokerCard::_5h};
std::vector<PokerCard> fullHouseA    = {PokerCard::Ac, PokerCard::Ad, PokerCard::Ah, PokerCard::_7s, PokerCard::_7h};

void testHandType(const std::vector<PokerCard>& cards,
                  PokerHandType                 expectedType,
                  PokerHandType (*getHandTypeFunc)(const std::vector<PokerCard>&)) {
  PokerHand     hand(cards);
  PokerHandType actualType = getHandTypeFunc(cards);
  std::cout << "Testing hand: " << hand.toString() << "\n";
  std::cout << "Expected: " << Poker::toString(expectedType) << ", Got: " << Poker::toString(actualType) << "\n";
  if (actualType == expectedType) {
    std::cout << "\033[32mTest passed.\033[0m\n";  // Green text for passed tests
  } else {
    std::cout << "\033[31mTest failed.\033[0m\n";  // Red text for failed tests
  }
  std::cout << "-----------------------------\n";
}