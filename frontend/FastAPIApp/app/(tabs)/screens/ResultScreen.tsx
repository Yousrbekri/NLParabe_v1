import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import { useLocalSearchParams } from 'expo-router';

export default function ResultScreen() {
  const { sentiment } = useLocalSearchParams();

  // Définir une icône en fonction du sentiment
  const sentimentIcon = sentiment === 'positif' ? '👍' : '👎';

  return (
    <View style={styles.container}>
      <View style={styles.card}>
        <Text style={styles.icon}>{sentimentIcon}</Text>
        <Text style={styles.resultText}>Sentiment : {sentiment}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f4f8', // Fond légèrement gris
  },
  card: {
    backgroundColor: '#fff', // Fond blanc
    borderRadius: 20, // Bordures arrondies
    padding: 30,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000', // Ombre
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5, // Ombre pour Android
    width: '80%', // Largeur de la carte
  },
  icon: {
    fontSize: 60, // Taille de l'icône
    marginBottom: 20, // Espacement sous l'icône
  },
  resultText: {
    fontSize: 24,
    color: '#333', // Texte gris foncé
    fontWeight: 'bold', // Texte en gras
    textTransform: 'capitalize', // Première lettre en majuscule
  },
});